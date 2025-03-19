import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.config import logger, DB_PATH, api_status

# ==============================================
# 4. SUBTITLE PROCESSING FOR VECTOR DATABASE
# ==============================================

def process_subtitles_to_documents(subtitles: List[Dict], video_info: Dict, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """Convert subtitles to Document objects for vector database
    
    Args:
        subtitles: List of subtitle dictionaries
        video_info: Dictionary with video information
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects
    """
    if not subtitles:
        return []
    
    # Combine adjacent subtitles into a single text
    full_text = ""
    timestamps = []
    
    for subtitle in subtitles:
        start_time = subtitle.get('start', 0)
        text = subtitle.get('text', '')
        
        full_text += text + " "
        timestamps.append((len(full_text) - len(text), start_time))
    
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split text into chunks
    texts = text_splitter.split_text(full_text)
    
    # Create Document objects with metadata
    documents = []
    for i, text_chunk in enumerate(texts):
        # Find the timestamps that fall within this chunk
        chunk_start_pos = i * (chunk_size - chunk_overlap) if i > 0 else 0
        chunk_end_pos = chunk_start_pos + len(text_chunk)
        
        # Find the closest timestamp for this chunk
        relevant_timestamps = [ts for pos, ts in timestamps if pos >= chunk_start_pos and pos <= chunk_end_pos]
        start_time = relevant_timestamps[0] if relevant_timestamps else 0
        
        # Calculate time in HH:MM:SS format
        minutes, seconds = divmod(int(start_time), 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Create document with metadata
        doc = Document(
            page_content=text_chunk,
            metadata={
                "video_id": video_info.get("video_id", ""),
                "title": video_info.get("title", "Unknown"),
                "author": video_info.get("author", "Unknown"),
                "language": video_info.get("language", "Unknown"),
                "timestamp": start_time,
                "time_str": time_str,
                "chunk_id": i
            }
        )
        documents.append(doc)
    
    return documents


def get_embedding_model(model_name: str = "huggingface"):
    """Get embedding model based on selection
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        Embedding model instance
    """
    # Handle dict input from Gradio dropdown
    if isinstance(model_name, dict):
        model_name = model_name.get("value", "huggingface")
    
    if model_name == "openai" and api_status["openai"]:
        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
    else:
        # Default to HuggingFace
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


def create_vector_db(documents: List[Document], embedding_model: str = "huggingface", video_id: str = None, video_info: dict = None):
    """Create or update a vector database with documents
    
    Args:
        documents: List of Document objects
        embedding_model: Name of the embedding model to use
        video_id: Video ID to use as collection name
        video_info: Video metadata to save
        
    Returns:
        Chroma vector database instance
    """
    embeddings = get_embedding_model(embedding_model)
    
    collection_name = f"video_{video_id}" if video_id else "subtitles"
    
    # Create vector database
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=DB_PATH
    )
    
    # Save metadata if provided
    if video_id and video_info:
        save_video_metadata(video_id, video_info)
    
    logger.info(f"Created vector database with {len(documents)} documents")
    
    return vectordb

def get_existing_vector_db(video_id: str, embedding_model: str = "huggingface"):
    """Get existing vector database for a video
    
    Args:
        video_id: Video ID to use as collection name
        embedding_model: Name of the embedding model to use
        
    Returns:
        Chroma vector database instance or None if not found
    """
    embeddings = get_embedding_model(embedding_model)
    collection_name = f"video_{video_id}"
    
    try:
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=DB_PATH
        )
        
        # Check if the collection exists
        if vectordb._collection.count() > 0:
            logger.info(f"Found existing vector database for video {video_id}")
            return vectordb
        else:
            logger.info(f"No existing vector database found for video {video_id}")
            return None
    except Exception as e:
        logger.warning(f"Error accessing vector database: {e}")
        return None

# Функция для загрузки существующей базы данных по ID видео
def load_database_by_id(video_id, embedding_model="huggingface"):
    """
    Загружает существующую векторную базу данных по ID видео
    
    Args:
        video_id (str): ID видео
        embedding_model (str): Модель эмбеддингов
        
    Returns:
        tuple: (status, vectordb, video_info)
    """
    import os
    import json
    
    collection_name = f"video_{video_id}"
    collection_path = os.path.join(DB_PATH, collection_name)
    
    # Проверяем существование коллекции
    if not os.path.exists(collection_path):
        return (False, None, None, f"База данных для видео {video_id} не найдена")
    
    # Пытаемся загрузить метаданные
    metadata_path = os.path.join(collection_path, "metadata.json")
    video_info = None
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                video_info = json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить метаданные для {video_id}: {e}")
    
    # Если метаданные не найдены, создаем базовую информацию
    if not video_info:
        video_info = {
            "video_id": video_id,
            "title": f"Видео {video_id}",
            "language": "unknown"
        }
    
    try:
        # Загружаем векторную базу данных
        embeddings = get_embedding_model(embedding_model)
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=DB_PATH
        )
        
        # Получаем примерное содержание (если доступно)
        sample_content = None
        try:
            # Получаем несколько документов для отображения содержимого
            results = vectordb.similarity_search("", k=3)
            if results:
                sample_content = "\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            logger.warning(f"Не удалось получить примерное содержание: {e}")
        
        return (True, vectordb, video_info, sample_content)
    
    except Exception as e:
        logger.error(f"Ошибка загрузки базы данных: {e}")
        return (False, None, None, f"Ошибка загрузки базы данных: {e}")


# Функция для сохранения метаданных видео
def save_video_metadata(video_id, video_info):
    """
    Сохраняет метаданные видео в файл JSON
    
    Args:
        video_id (str): ID видео
        video_info (dict): Информация о видео
    """
    import os
    import json
    from datetime import datetime
    
    collection_name = f"video_{video_id}"
    collection_path = os.path.join(DB_PATH, collection_name)
    
    # Создаем директорию, если она не существует
    os.makedirs(collection_path, exist_ok=True)
    
    # Добавляем дату создания
    metadata = video_info.copy()
    metadata["created_at"] = datetime.now().isoformat()
    
    # Сохраняем метаданные в JSON файл
    metadata_path = os.path.join(collection_path, "metadata.json")
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Метаданные сохранены для {video_id}")
    except Exception as e:
        logger.error(f"Не удалось сохранить метаданные для {video_id}: {e}")


# Функция для получения списка всех сохраненных баз данных
def get_saved_databases():
    """
    Получает список всех сохраненных векторных баз данных
    
    Returns:
        list: Список словарей с информацией о сохраненных базах данных
    """
    import os
    import json
    
    saved_dbs = []
    
    # Проверяем существование директории
    if not os.path.exists(DB_PATH):
        logger.warning(f"Директория базы данных не найдена: {DB_PATH}")
        return saved_dbs
    
    # Ищем все коллекции (папки, начинающиеся с "video_")
    for item in os.listdir(DB_PATH):
        collection_path = os.path.join(DB_PATH, item)
        
        # Проверяем, что это директория и начинается с "video_"
        if os.path.isdir(collection_path) and item.startswith("video_"):
            video_id = item[6:]  # Убираем префикс "video_"
            
            # Пытаемся загрузить метаданные, если они существуют
            metadata_path = os.path.join(collection_path, "metadata.json")
            metadata = {}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить метаданные для {item}: {e}")
            
            # Получаем заголовок и дату создания
            title = metadata.get("title", "Неизвестное видео")
            created_at = metadata.get("created_at", "Неизвестная дата")
            
            saved_dbs.append({
                "collection_name": item,
                "video_id": video_id,
                "title": title,
                "created_at": created_at
            })
    
    # Сортируем по дате создания (последние сверху)
    saved_dbs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return saved_dbs
