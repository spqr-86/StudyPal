import logging
import nltk
from typing import List, Dict, Any
from tqdm.notebook import tqdm
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from src.config import logger, APP_SETTINGS
from src import app_state
from src.utils import format_time
from src.youtube import get_youtube_chapters, get_youtube_video_chapters_api


def analyze_subtitles_into_blocks(subtitles, min_block_duration=60, min_pause_threshold=3, max_block_size=25):
    """
    Разбивает субтитры на логические блоки на основе контента и временных меток
    
    Args:
        subtitles: Список словарей субтитров
        min_block_duration: Минимальная продолжительность блока в секундах
        min_pause_threshold: Минимальный порог паузы для разделения блоков в секундах
        max_block_size: Максимальное количество субтитров в одном блоке
        
    Returns:
        Список блоков, где каждый блок содержит список субтитров и метаданные
    """
    if not subtitles or len(subtitles) < 5:
        # Если субтитров мало, возвращаем один блок
        return [{
            "start_time": subtitles[0]["start"] if subtitles else 0,
            "end_time": subtitles[-1]["start"] + subtitles[-1].get("duration", 5) if subtitles else 0,
            "subtitles": subtitles,
            "content_text": " ".join([s.get("text", "") for s in subtitles]),
            "title": "Весь контент"
        }]
    
    blocks = []
    current_block = []
    current_block_text = ""
    
    for i, subtitle in enumerate(subtitles):
        # Добавляем текущий субтитр в текущий блок
        current_block.append(subtitle)
        current_block_text += " " + subtitle.get("text", "")
        
        # Проверяем, нужно ли начать новый блок
        should_split = False
        
        # Проверка на основе паузы
        if i < len(subtitles) - 1:
            next_subtitle = subtitles[i + 1]
            pause_duration = next_subtitle["start"] - (subtitle["start"] + subtitle.get("duration", 5))
            if pause_duration >= min_pause_threshold:
                should_split = True
                
        # Проверка на основе размера блока
        if len(current_block) >= max_block_size:
            should_split = True
            
        # Создаем новый блок, если нужно разделить
        if should_split or i == len(subtitles) - 1:
            if current_block:
                block_start_time = current_block[0]["start"]
                block_end_time = current_block[-1]["start"] + current_block[-1].get("duration", 5)
                
                # Проверяем минимальную продолжительность блока
                if block_end_time - block_start_time >= min_block_duration or i == len(subtitles) - 1:
                    blocks.append({
                        "start_time": block_start_time,
                        "end_time": block_end_time,
                        "subtitles": current_block.copy(),
                        "content_text": current_block_text.strip(),
                        "title": ""  # Будет заполнено позже
                    })
                    current_block = []
                    current_block_text = ""
    
    # Заполняем заголовки блоков
    blocks = generate_block_titles(blocks, method="enhanced_keywords")
    
    return blocks


# Модифицированная функция для создания блоков с учетом существующих глав
def analyze_subtitles_into_blocks_with_chapters(subtitles, video_id, video_info=None, min_block_duration=60):
    """
    Разбивает субтитры на логические блоки с учетом существующих глав видео
    
    Args:
        subtitles: Список словарей субтитров
        video_id: ID видео YouTube
        video_info: Информация о видео (опционально)
        min_block_duration: Минимальная продолжительность блока в секундах
        
    Returns:
        Список блоков, где каждый блок содержит список субтитров и метаданные
    """
    if not subtitles:
        return []
    
    # Пытаемся получить главы видео
    chapters = get_youtube_chapters(video_id)
    
    # Если главы не найдены через парсинг, пробуем через API (если есть ключ)
    if not chapters:
        chapters = get_youtube_video_chapters_api(video_id)
    
    # Определяем конечное время видео
    video_end_time = 0
    if subtitles:
        last_subtitle = subtitles[-1]
        video_end_time = last_subtitle["start"] + last_subtitle.get("duration", 5)
    
    # Если главы найдены, используем их для создания блоков
    if chapters:
        logger.info(f"Using {len(chapters)} chapters from YouTube for blocks")
        
        # Заполняем отсутствующие времена окончания глав
        for i, chapter in enumerate(chapters):
            if chapter["end_time"] is None:
                if i < len(chapters) - 1:
                    chapter["end_time"] = chapters[i + 1]["start_time"]
                else:
                    chapter["end_time"] = video_end_time
        
        # Создаем блоки на основе глав
        blocks = []
        for chapter in chapters:
            start_time = chapter["start_time"]
            end_time = chapter["end_time"]
            
            # Фильтруем субтитры для этой главы
            chapter_subtitles = [
                s for s in subtitles 
                if s["start"] >= start_time and s["start"] < end_time
            ]
            
            # Если есть субтитры или продолжительность достаточная, создаем блок
            if chapter_subtitles or (end_time - start_time) >= min_block_duration:
                content_text = " ".join([s.get("text", "") for s in chapter_subtitles])
                
                blocks.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "subtitles": chapter_subtitles,
                    "content_text": content_text,
                    "title": chapter["title"]  # Используем заголовок из YouTube
                })
        
        return blocks
    
    # Если главы не найдены, используем стандартный алгоритм
    logger.info("No YouTube chapters found, using automatic block detection")
    return analyze_subtitles_into_blocks(subtitles, min_block_duration)


def generate_block_titles(blocks, method="enhanced_keywords"):
    """
    Генерирует содержательные заголовки для блоков субтитров
    
    Args:
        blocks: Список блоков субтитров
        method: Метод генерации заголовков ('enhanced_keywords', 'openai', 'first_sentence')
        
    Returns:
        Список блоков с заполненными заголовками
    """
    if method == "enhanced_keywords":
        # Улучшенный метод на основе ключевых слов и первых предложений
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize, sent_tokenize
            from collections import Counter
            
            # Загрузка необходимых ресурсов NLTK
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            # Получаем стоп-слова для соответствующих языков
            try:
                stop_words_en = set(stopwords.words('english'))
            except:
                stop_words_en = set()
            
            try:
                stop_words_ru = set(stopwords.words('russian'))
            except:
                stop_words_ru = set()
            
            # Объединяем стоп-слова разных языков
            stop_words = stop_words_en.union(stop_words_ru)
            
            # Добавляем дополнительные общие стоп-слова
            additional_stop_words = {'yeah', 'uh', 'um', 'oh', 'like', 'just', 'so', 'know', 'think', 'well', 'going', 
                                    'get', 'got', 'actually', 'okay', 'right', 'thing', 'things', 'gonna', 'wanna'}
            stop_words = stop_words.union(additional_stop_words)
            
            for i, block in enumerate(blocks):
                text = block["content_text"].lower()
                
                # Получаем первое предложение
                sentences = sent_tokenize(text)
                first_sentence = sentences[0] if sentences else ""
                
                # Очищаем первое предложение (берем не более 7 слов)
                clean_first_words = []
                for word in first_sentence.split()[:7]:
                    # Очищаем от пунктуации
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if clean_word and len(clean_word) > 1:
                        clean_first_words.append(clean_word)
                
                first_phrase = " ".join(clean_first_words)
                
                # Находим ключевые слова из всего блока
                words = word_tokenize(text)
                words = [word.lower() for word in words 
                         if word.isalnum() and word.lower() not in stop_words and len(word) > 2]
                
                word_counts = Counter(words)
                top_words = [word for word, count in word_counts.most_common(4) if count > 1]
                
                # Формируем заголовок
                if top_words and first_phrase:
                    # Используем первую фразу и топ-ключевые слова
                    if len(first_phrase) > 30:
                        first_phrase = first_phrase[:30] + "..."
                    
                    key_words = ", ".join(top_words[:3]) if top_words else ""
                    
                    # Объединяем части в заголовок
                    title = first_phrase.capitalize()
                    if key_words:
                        title += f" [{key_words}]"
                elif first_phrase:
                    # Используем только первую фразу
                    title = first_phrase.capitalize()
                elif top_words:
                    # Используем только ключевые слова
                    title = "Topic: " + ", ".join(top_words[:4])
                else:
                    # Если не удалось извлечь полезную информацию
                    title = f"Section {i+1}"
                
                # Ограничиваем длину заголовка
                if len(title) > 70:
                    title = title[:70] + "..."
                
                block["title"] = title
                
        except Exception as e:
            logger.warning(f"Error generating enhanced keyword titles: {e}")
            # Fallback: простая нумерация разделов с первыми словами
            for i, block in enumerate(blocks):
                try:
                    text = block["content_text"]
                    first_words = " ".join(text.split()[:5])
                    block["title"] = f"Section {i+1}: {first_words}..."
                except:
                    block["title"] = f"Section {i+1}"
    
    elif method == "openai" and api_status.get("openai", False):
        # Метод с использованием OpenAI для генерации заголовков
        try:
            from langchain_openai import ChatOpenAI
            
            model = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3
            )
            
            for block in tqdm(blocks, desc="Generating titles with OpenAI"):
                content = block["content_text"]
                
                # Ограничиваем длину текста для API
                if len(content) > 2000:
                    content = content[:2000] + "..."
                
                try:
                    response = model.invoke(
                        f"Generate a concise, informative title (5-10 words) for this text segment from a video: '{content}'"
                    )
                    title = response.content.strip().strip('"\'')
                    block["title"] = title
                except Exception as e:
                    logger.warning(f"Error generating title with OpenAI: {e}")
                    # Fallback to simpler method
                    first_words = " ".join(content.split()[:7])
                    block["title"] = first_words + "..."
        except Exception as e:
            logger.error(f"Failed to use OpenAI for title generation: {e}")
            # Fallback to keywords method
            blocks = generate_block_titles(blocks, method="enhanced_keywords")
    
    elif method == "first_sentence":
        # Простой метод на основе первого предложения
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # Загрузка необходимых ресурсов NLTK
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            for i, block in enumerate(blocks):
                text = block["content_text"]
                
                # Получаем первое предложение
                sentences = sent_tokenize(text)
                if sentences:
                    first_sent = sentences[0]
                    # Ограничиваем длину
                    if len(first_sent) > 60:
                        title = first_sent[:60] + "..."
                    else:
                        title = first_sent
                    block["title"] = title
                else:
                    # Если не удалось разбить на предложения
                    first_words = " ".join(text.split()[:7])
                    block["title"] = f"Section {i+1}: {first_words}..."
        except Exception as e:
            logger.warning(f"Error generating first sentence titles: {e}")
            # Fallback: простая нумерация разделов с первыми словами
            for i, block in enumerate(blocks):
                text = block["content_text"]
                first_words = " ".join(text.split()[:5])
                block["title"] = f"Section {i+1}: {first_words}..."
    
    else:
        # Простой метод на основе первых слов
        for i, block in enumerate(blocks):
            try:
                content = block["content_text"]
                first_words = " ".join(content.split()[:7])
                if first_words:
                    block["title"] = first_words.capitalize() + "..."
                else:
                    block["title"] = f"Section {i+1}"
            except Exception as e:
                logger.warning(f"Error generating simple title for block {i}: {e}")
                block["title"] = f"Section {i+1}"
    
    return blocks



# Обновление функции generate_table_of_contents для поддержки YouTube глав
def generate_table_of_contents(blocks):
    """
    Генерирует оглавление на основе блоков субтитров с улучшенным форматированием
    
    Args:
        blocks: Список блоков субтитров
        
    Returns:
        Строка в формате Markdown с оглавлением
    """
    toc = "# Оглавление видео\n\n"
    
    # Проверяем, есть ли блоки с флагом YouTube глав
    has_youtube_chapters = any(block.get("is_youtube_chapter", False) for block in blocks)
    
    if has_youtube_chapters:
        toc += "> ℹ️ Оглавление создано на основе глав YouTube\n\n"
    
    for i, block in enumerate(blocks):
        # Форматируем временные метки
        start_time = format_time(block["start_time"])
        duration = format_time(block["end_time"] - block["start_time"])
        
        # Полный заголовок без сокращений
        title = block['title']
        
        # Добавляем значок для глав YouTube
        chapter_icon = "🔖 " if block.get("is_youtube_chapter", False) else ""
        
        # Форматируем пункт оглавления с переносом строки для улучшения читаемости
        toc += f"### {i+1}. {chapter_icon}{title}\n"
        toc += f"**Время:** {start_time} | **Длительность:** {duration}\n\n"
    
    return toc

# Функция для поиска доступных источников глав
def check_chapter_sources(video_id):
    """
    Проверяет доступные источники глав для видео
    
    Args:
        video_id: ID видео YouTube
        
    Returns:
        dict: Словарь с информацией о доступных источниках глав
    """
    result = {
        "has_chapters": False,
        "sources": []
    }
    
    # Проверяем наличие глав через парсинг
    try:
        chapters = get_youtube_chapters(video_id)
        if chapters:
            result["has_chapters"] = True
            result["sources"].append("youtube_html")
            result["chapters_count"] = len(chapters)
    except Exception as e:
        logger.warning(f"Error checking HTML chapters: {e}")
    
    # Проверяем наличие глав через API
    try:
        api_chapters = get_youtube_video_chapters_api(video_id)
        if api_chapters:
            result["has_chapters"] = True
            result["sources"].append("youtube_api")
            result["api_chapters_count"] = len(api_chapters)
    except Exception as e:
        logger.warning(f"Error checking API chapters: {e}")
    
    return result


def display_toc_entry(block_index):
    """
    Отображает полную информацию о выбранном блоке для оглавления
    
    Args:
        block_index: Индекс блока
        
    Returns:
        Строка в формате Markdown с детальной информацией о блоке
    """
    if not hasattr(app_state, 'subtitle_blocks') or not app_state.subtitle_blocks:
        return "Блоки субтитров не найдены. Сначала обработайте видео."
    
    try:
        block_index = int(block_index)
    except (ValueError, TypeError):
        return f"Неверный индекс блока: '{block_index}'. Должно быть целое число."
    
    if block_index < 0 or block_index >= len(app_state.subtitle_blocks):
        return f"Ошибка: индекс блока {block_index} вне диапазона (0-{len(app_state.subtitle_blocks)-1})"
    
    try:
        block = app_state.subtitle_blocks[block_index]
        
        # Создаем подробное описание блока
        content = f"## {block.get('title', f'Блок {block_index+1}')}\n\n"
        
        # Добавляем временные метки
        if "start_time" in block and "end_time" in block:
            start_time = format_time(block['start_time'])
            end_time = format_time(block['end_time'])
            duration = format_time(block['end_time'] - block['start_time'])
            
            content += f"**Начало:** {start_time} | **Конец:** {end_time} | **Длительность:** {duration}\n\n"
        
        # Добавляем краткий обзор содержимого
        # Берем первые 100-200 символов текста для предпросмотра
        if "content_text" in block and block["content_text"]:
            preview_text = block["content_text"]
            if len(preview_text) > 200:
                preview_text = preview_text[:200] + "..."
            
            content += f"**Обзор содержимого:**\n\n{preview_text}\n\n"
        
        # Добавляем количество субтитров в блоке
        if "subtitles" in block and block["subtitles"]:
            content += f"**Количество субтитров в блоке:** {len(block['subtitles'])}\n\n"
        
        content += "Нажмите кнопку 'Показать содержимое блока' для просмотра полного текста."
        
        return content
    except Exception as e:
        logger.error(f"Error displaying TOC entry: {e}")
        return f"Ошибка при отображении информации о блоке: {str(e)}"


# Обновленная функция process_subtitles_with_blocks для поддержки глав
def process_subtitles_with_blocks(subtitles, video_info):
    """
    Обрабатывает субтитры, разбивая их на логические блоки и генерируя оглавление
    
    Args:
        subtitles: Список словарей субтитров
        video_info: Словарь с информацией о видео
        
    Returns:
        Tuple of (blocks, table_of_contents)
    """
    video_id = video_info.get("video_id") if video_info else None
    
    if video_id:
        # Попытка разбить субтитры с учетом существующих глав
        blocks = analyze_subtitles_into_blocks_with_chapters(subtitles, video_id, video_info)
    else:
        # Стандартное разбиение на блоки
        blocks = analyze_subtitles_into_blocks(subtitles)
    
    # Если заголовки не были заданы из YouTube глав, генерируем их
    for block in blocks:
        if not block["title"] or block["title"].startswith("Section "):
            # Блок не имеет заголовка из YouTube, генерируем
            block["title"] = ""  # Сбрасываем заголовок
    
    # Блоки, которым нужны заголовки
    blocks_to_title = [b for b in blocks if not b["title"]]
    if blocks_to_title:
        titled_blocks = generate_block_titles(blocks_to_title, method="enhanced_keywords")
        
        # Заменяем блоки без заголовков на блоки с заголовками
        for i, block in enumerate(blocks):
            if not block["title"]:
                # Ищем соответствующий блок с заголовком
                for titled_block in titled_blocks:
                    if titled_block["start_time"] == block["start_time"] and titled_block["end_time"] == block["end_time"]:
                        block["title"] = titled_block["title"]
                        break
    
    # Генерируем оглавление
    toc = generate_table_of_contents(blocks)
    
    # Сохраняем блоки и оглавление в состоянии приложения
    app_state.subtitle_blocks = blocks
    app_state.table_of_contents = toc
    
    # Сохраняем информацию о блоках в метаданных видео
    if video_id:
        try:
            # Сохраняем только основную информацию о блоках (без полных субтитров)
            blocks_metadata = []
            for block in blocks:
                blocks_metadata.append({
                    "start_time": block["start_time"],
                    "end_time": block["end_time"],
                    "title": block["title"],
                    "is_youtube_chapter": block.get("is_youtube_chapter", False)
                })
            
            # Добавляем информацию о блоках в метаданные видео
            video_info["blocks"] = blocks_metadata
            video_info["has_youtube_chapters"] = any(b.get("is_youtube_chapter", False) for b in blocks)
            
            # Сохраняем обновленные метаданные
            save_video_metadata(video_id, video_info)
        except Exception as e:
            logger.warning(f"Failed to save blocks metadata: {e}")
    
    return blocks, toc

# Исправление функции get_block_content для обработки ошибок
def get_block_content(block_index):
    """
    Получает форматированное содержимое блока по его индексу
    
    Args:
        block_index: Индекс блока
        
    Returns:
        Строка в формате Markdown с содержимым блока
    """
    if not hasattr(app_state, 'subtitle_blocks') or not app_state.subtitle_blocks:
        return "Блоки субтитров не найдены. Сначала обработайте видео."
    
    try:
        block_index = int(block_index)
    except (ValueError, TypeError):
        return f"Неверный индекс блока: '{block_index}'. Должно быть целое число."
    
    if block_index < 0 or block_index >= len(app_state.subtitle_blocks):
        return f"Ошибка: индекс блока {block_index} вне диапазона (0-{len(app_state.subtitle_blocks)-1})"
    
    try:
        block = app_state.subtitle_blocks[block_index]
        
        # Форматируем содержимое блока
        content = f"## {block.get('title', f'Блок {block_index+1}')}\n\n"
        
        # Проверяем наличие временных меток
        if "start_time" in block and "end_time" in block:
            content += f"**Временная метка:** {format_time(block['start_time'])} - {format_time(block['end_time'])}\n\n"
        
        content += "### Содержание:\n\n"
        
        # Добавляем субтитры блока с временными метками
        if "subtitles" in block and block["subtitles"]:
            for subtitle in block["subtitles"]:
                if "start" in subtitle:
                    timestamp = format_time(subtitle["start"])
                    text = subtitle.get("text", "")
                    content += f"**[{timestamp}]** {text}\n\n"
        else:
            # Если субтитры отсутствуют, показываем основной текст блока
            content += block.get("content_text", "Содержимое недоступно.")
        
        return content
    except Exception as e:
        logger.error(f"Error getting block content: {e}")
        return f"Ошибка при получении содержимого блока: {str(e)}"

