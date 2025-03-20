import os
import re
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from bs4 import BeautifulSoup
from src.config import logger


def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract video ID from YouTube URL
    
    Args:
        youtube_url: URL of the YouTube video
        
    Returns:
        Video ID or None if not found
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embedded URLs
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',  # Watch URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Shortened youtu.be URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    return None


# Обновление функции get_youtube_subtitles для получения информации о главах
def get_youtube_subtitles(youtube_url, languages=['en']):
    """
    Get subtitles from a YouTube video using youtube-transcript-api
    
    Args:
        youtube_url (str): URL of the YouTube video
        languages (list): List of preferred language codes
        
    Returns:
        dict: Dictionary containing subtitles and video information
    """
    video_id = extract_video_id(youtube_url)
    
    if not video_id:
        logger.error(f"Invalid YouTube URL: {youtube_url}")
        return {
            "success": False,
            "error": "Invalid YouTube URL",
            "subtitles": [],
            "video_info": {"video_id": None}
        }
    
    # Ensure languages are strings, not dicts (for compatibility with Gradio)
    if languages and isinstance(languages[0], dict):
        languages = [lang.get("value", "en") if isinstance(lang, dict) else lang for lang in languages]
    
    # Get basic video info
    video_info = {
        "video_id": video_id,
        "url": youtube_url
    }
    
    try:
        # Get video metadata (title, etc.)
        try:
            import urllib.request
            
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = urllib.request.urlopen(oembed_url)
            data = json.loads(response.read())
            
            video_info.update({
                "title": data.get("title", "Unknown"),
                "author": data.get("author_name", "Unknown"),
                "thumbnail": data.get("thumbnail_url", "")
            })
        except Exception as e:
            logger.warning(f"Could not get video metadata: {e}")
            video_info.update({
                "title": "Unknown title",
                "author": "Unknown author",
                "thumbnail": ""
            })
        
        # Try to get chapters information
        chapters = get_youtube_chapters(video_id)
        
        # If chapters not found through parsing, try through API
        if not chapters:
            chapters = get_youtube_video_chapters_api(video_id)
        
        if chapters:
            video_info["chapters"] = chapters
            video_info["has_chapters"] = True
            logger.info(f"Found {len(chapters)} chapters for video {video_id}")
        
        # Try to get transcript in any of the preferred languages
        available_transcripts = []
        
        # First try to get manually created transcripts in the preferred languages
        for language in languages:
            try:
                # This will raise an exception if no transcript is found
                transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
                
                # If we get here, transcript was found
                video_info["language"] = language
                video_info["language_code"] = language
                video_info["is_generated"] = False  # We don't know, but assume not generated
                
                logger.info(f"Found subtitles in {language}")
                
                return {
                    "success": True,
                    "subtitles": transcripts,
                    "video_info": video_info
                }
            except Exception as e:
                logger.debug(f"No transcript in {language}: {e}")
                continue
        
        # If no transcript is found in preferred languages, get any available one
        try:
            # Just try to get any transcript available
            transcripts = YouTubeTranscriptApi.get_transcript(video_id)
            
            # If we get here, transcript was found
            video_info["language"] = "Unknown"  # We don't know which language was selected
            video_info["language_code"] = "unknown"
            
            logger.info(f"Found subtitles (language unknown)")
            
            return {
                "success": True,
                "subtitles": transcripts,
                "video_info": video_info
            }
        except Exception as e:
            logger.error(f"Could not find any subtitles: {e}")
            raise NoTranscriptFound("No transcripts available for this video")
            
    except Exception as e:
        logger.error(f"Error extracting subtitles: {str(e)}")
        return {
            "success": False,
            "error": f"Error extracting subtitles: {str(e)}",
            "subtitles": [],
            "video_info": video_info
        }


# Функция для получения информации о главах YouTube видео
def get_youtube_chapters(video_id):
    """
    Получает информацию о главах (chapters) YouTube видео, если они доступны
    
    Args:
        video_id: ID видео YouTube
        
    Returns:
        list: Список глав с временными метками, или пустой список, если главы недоступны
    """
    import requests
    import re
    from bs4 import BeautifulSoup
    
    try:
        # Запрашиваем страницу видео
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.warning(f"Failed to get video page: HTTP {response.status_code}")
            return []
        
        # Используем BeautifulSoup для парсинга HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Ищем скрипты с данными
        scripts = soup.find_all('script')
        chapters_data = []
        
        # Поиск информации о главах в скриптах
        for script in scripts:
            script_text = script.string
            if script_text and "chapters" in script_text:
                # Ищем паттерн с данными о главах
                chapter_pattern = re.compile(r'\"chapters\":\[(.*?)\]')
                matches = chapter_pattern.findall(script_text)
                
                if matches:
                    # Парсим данные о главах
                    import json
                    try:
                        # Пытаемся преобразовать в JSON
                        chapters_json = f"[{matches[0]}]"
                        chapters_data = json.loads(chapters_json)
                        logger.info(f"Found {len(chapters_data)} chapters in video {video_id}")
                        break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse chapters JSON: {matches[0]}")
        
        if not chapters_data:
            # Альтернативный метод: поиск в HTML
            chapter_elements = soup.select('div.ytp-chapter-title-content')
            if chapter_elements:
                for element in chapter_elements:
                    # Парсим заголовок и время
                    title = element.text.strip()
                    time_elem = element.find_previous('span', class_='ytp-time-current')
                    if time_elem and title:
                        time_str = time_elem.text
                        # Преобразуем время в секунды
                        h, m, s = 0, 0, 0
                        if ':' in time_str:
                            parts = time_str.split(':')
                            if len(parts) == 2:
                                m, s = map(int, parts)
                            elif len(parts) == 3:
                                h, m, s = map(int, parts)
                        
                        seconds = h * 3600 + m * 60 + s
                        chapters_data.append({
                            "title": title,
                            "start_time": seconds
                        })
        
        # Преобразуем данные в стандартный формат
        chapters = []
        for i, chapter in enumerate(chapters_data):
            # Получаем время начала текущей главы
            start_time = chapter.get("start_time") or chapter.get("startTime") or 0
            
            # Получаем время окончания (начало следующей главы или конец видео)
            end_time = None
            if i < len(chapters_data) - 1:
                next_chapter = chapters_data[i + 1]
                end_time = next_chapter.get("start_time") or next_chapter.get("startTime") or None
            
            # Получаем заголовок
            title = chapter.get("title") or chapter.get("chapterName") or f"Chapter {i+1}"
            
            chapters.append({
                "title": title,
                "start_time": start_time,
                "end_time": end_time
            })
        
        return chapters
        
    except Exception as e:
        logger.error(f"Error fetching YouTube chapters: {e}")
        return []


# Функция для получения заголовков YouTube с API Data
def get_youtube_video_chapters_api(video_id, api_key=None):
    """
    Получает информацию о главах видео через YouTube Data API
    
    Args:
        video_id: ID видео
        api_key: API ключ YouTube Data API (опционально)
        
    Returns:
        list: Список глав или пустой список
    """
    if not api_key:
        # Если API ключ не предоставлен, пытаемся получить его из окружения
        api_key = os.getenv('YOUTUBE_DATA_API_KEY')
    
    if not api_key:
        logger.warning("No YouTube Data API key provided")
        return []
    
    try:
        import requests
        
        # Запрашиваем информацию о видео через API
        url = f"https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet,contentDetails",
            "id": video_id,
            "key": api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            logger.warning(f"Failed to get video data: HTTP {response.status_code}")
            return []
        
        data = response.json()
        
        # Проверяем, есть ли информация о видео
        if "items" not in data or len(data["items"]) == 0:
            logger.warning(f"No video data found for {video_id}")
            return []
        
        # Получаем описание видео, которое может содержать временные метки
        description = data["items"][0]["snippet"].get("description", "")
        
        # Ищем временные метки в описании
        import re
        
        # Паттерн для поиска временных меток (например, "00:30 Заголовок" или "1:45:30 Заголовок")
        timestamp_pattern = re.compile(r'((?:\d{1,2}:)?\d{1,2}:\d{2})\s+(.*?)(?=\n(?:\d{1,2}:)?\d{1,2}:\d{2}|\n\n|$)', re.MULTILINE)
        matches = timestamp_pattern.findall(description)
        
        if not matches:
            logger.info(f"No timestamps found in description for {video_id}")
            return []
        
        # Преобразуем временные метки в секунды и создаем список глав
        chapters = []
        for i, (time_str, title) in enumerate(matches):
            # Преобразуем время в секунды
            h, m, s = 0, 0, 0
            parts = time_str.split(':')
            if len(parts) == 2:
                m, s = map(int, parts)
            elif len(parts) == 3:
                h, m, s = map(int, parts)
            
            start_time = h * 3600 + m * 60 + s
            
            # Определяем время окончания (начало следующей главы или None)
            end_time = None
            if i < len(matches) - 1:
                next_time_str = matches[i + 1][0]
                next_parts = next_time_str.split(':')
                next_h, next_m, next_s = 0, 0, 0
                if len(next_parts) == 2:
                    next_m, next_s = map(int, next_parts)
                elif len(next_parts) == 3:
                    next_h, next_m, next_s = map(int, next_parts)
                
                end_time = next_h * 3600 + next_m * 60 + next_s
            
            chapters.append({
                "title": title.strip(),
                "start_time": start_time,
                "end_time": end_time
            })
        
        logger.info(f"Found {len(chapters)} chapters in description for {video_id}")
        return chapters
        
    except Exception as e:
        logger.error(f"Error fetching chapters from YouTube API: {e}")
        return []


# Function to format subtitles with timestamps
def format_subtitles(subtitles):
    """Format subtitles with timestamps
    
    Args:
        subtitles: List of subtitle dictionaries
        
    Returns:
        Formatted text with timestamps
    """
    if not subtitles:
        return "No subtitles available."
    
    formatted_text = ""
    
    for entry in subtitles:
        start_seconds = entry.get('start', 0)
        minutes, seconds = divmod(int(start_seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
        formatted_text += f"{timestamp} {entry.get('text', '')}\n\n"
    
    return formatted_text