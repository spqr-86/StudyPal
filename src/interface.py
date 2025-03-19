import gradio as gr
import logging
from typing import List, Dict, Any
from src.config import logger, AVAILABLE_LANGUAGES, TRANSLATION_LANGUAGES, CHAT_MODELS, EMBEDDING_MODELS
from src import app_state
from src.utils import display_info, format_time
from src.youtube import extract_video_id, get_youtube_subtitles
from src.processing import get_embedding_model, create_vector_db, get_existing_vector_db, get_saved_databases, load_database_by_id
from src.blocks import process_subtitles_with_blocks, display_toc_entry, get_block_content
from src.chat import setup_qa_chain, chat_with_subtitles
from src.translation import translate_subtitle_text


# Обновление функции process_video для поддержки разбиения на блоки
def process_video(youtube_url: str, embedding_model: str = "huggingface", language: str = "en"):
    """Process YouTube video to extract and store subtitles
    
    Args:
        youtube_url: URL of the YouTube video
        embedding_model: Name of the embedding model to use
        language: Preferred subtitle language
        
    Returns:
        Tuple of (status_html, video_info_html, subtitles_markdown)
    """
    try:
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return (
                "❌ Invalid YouTube URL. Please provide a valid URL.",
                "",
                ""
            )
        
        # Make sure language is a string, not a dict
        if isinstance(language, dict):
            language = language.get("value", "en")
        
        # Check if we already have processed this video
        existing_db = get_existing_vector_db(video_id, embedding_model)
        if existing_db:
            app_state.vectordb = existing_db
            
            # We need to retrieve the video info and subtitles
            # This could be stored in the database metadata, but for simplicity we'll re-fetch
            result = get_youtube_subtitles(youtube_url, [language])
            
            if result["success"]:
                subtitles = result["subtitles"]
                video_info = result["video_info"]
                
                app_state.subtitles = subtitles
                app_state.video_info = video_info
                
                # Разбиваем субтитры на блоки
                blocks, toc = process_subtitles_with_blocks(subtitles, video_info)
# Setup QA chain
                app_state.qa_chain = setup_qa_chain(app_state.vectordb, "huggingface")
                
                # Create formatted outputs
                video_info_html = f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <img src="{video_info.get('thumbnail', '')}" style="width: 120px; margin-right: 15px;">
                    <div>
                        <h3>{video_info.get('title', 'Unknown title')}</h3>
                        <p>By: {video_info.get('author', 'Unknown')}</p>
                        <p>Language: {video_info.get('language', 'Unknown')}</p>
                        <p>Blocks: {len(blocks)}</p>
                    </div>
                </div>
                """
                
                subtitles_markdown = format_subtitles(subtitles)
                
                return (
                    f"✅ Video already processed. Loaded existing data for video ID: {video_id}",
                    video_info_html,
                    subtitles_markdown
                )
            else:
                return (
                    f"❌ Failed to extract subtitles: {result.get('error', 'Unknown error')}",
                    "",
                    ""
                )
        
        # Extract subtitles
        result = get_youtube_subtitles(youtube_url, [language])
        
        if not result["success"]:
            error_msg = result.get("error", "Unknown error")
            return (
                f"❌ Failed to extract subtitles: {error_msg}",
                "",
                ""
            )
        
        subtitles = result["subtitles"]
        video_info = result["video_info"]
        
        # Store state
        app_state.subtitles = subtitles
        app_state.video_info = video_info
        
        # Разбиваем субтитры на блоки
        blocks, toc = process_subtitles_with_blocks(subtitles, video_info)
        
        # Process subtitles into documents
        documents = process_subtitles_to_documents(subtitles, video_info)
        
        # Create vector database with metadata
        app_state.vectordb = create_vector_db(documents, embedding_model, video_id, video_info)
        
        # Setup QA chain
        app_state.qa_chain = setup_qa_chain(app_state.vectordb, "huggingface")
        
        # Create formatted outputs
        video_info_html = f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="{video_info.get('thumbnail', '')}" style="width: 120px; margin-right: 15px;">
            <div>
                <h3>{video_info.get('title', 'Unknown title')}</h3>
                <p>By: {video_info.get('author', 'Unknown')}</p>
                <p>Language: {video_info.get('language', 'Unknown')}</p>
                <p>Blocks: {len(blocks)}</p>
            </div>
        </div>
        """
        
        subtitles_markdown = format_subtitles(subtitles)
        
        return (
            f"✅ Successfully processed video with ID: {video_id}",
            video_info_html,
            subtitles_markdown
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return (
            f"❌ Error: {str(e)}",
            "",
            ""
        )


# Обновление функции загрузки базы данных для поддержки блоков
def load_database_from_list(db_info):
    """
    Загружает базу данных из выбранного элемента в списке
    
    Args:
        db_info: Информация о базе данных (словарь или строка с video_id)
        
    Returns:
        Tuple of (status_html, video_info_html, subtitles_markdown)
    """
    try:
        # Проверяем тип входных данных
        if isinstance(db_info, dict):
            video_id = db_info.get("value")
        else:
            video_id = db_info
        
        if not video_id:
            return (
                "❌ Некорректный ID видео.",
                "",
                ""
            )
        
        # Загружаем базу данных
        success, vectordb, video_info, sample_content = load_database_by_id(video_id)
        
        if not success or not vectordb:
            return (
                f"❌ Не удалось загрузить базу данных: {sample_content}",
                "",
                ""
            )
        
        # Сохраняем базу данных в состояние приложения
        app_state.vectordb = vectordb
        app_state.video_info = video_info
        
        # Восстанавливаем субтитры из базы данных
        try:
            # Получаем несколько документов для восстановления субтитров
            results = vectordb.similarity_search("", k=50)
            subtitles = []
            
            for doc in results:
                timestamp = doc.metadata.get("timestamp")
                if timestamp is not None:
                    text = doc.page_content
                    
                    # Создаем запись субтитра
                    subtitle = {
                        "start": timestamp,
                        "duration": 5,  # Примерная длительность
                        "text": text
                    }
                    subtitles.append(subtitle)
            
            # Сортируем субтитры по времени начала
            subtitles.sort(key=lambda x: x["start"])
            
            # Сохраняем субтитры в состояние приложения
            app_state.subtitles = subtitles
        except Exception as e:
            logger.warning(f"Не удалось восстановить субтитры: {e}")
            app_state.subtitles = []
        
        # Проверяем наличие информации о блоках в метаданных
        if "blocks" in video_info:
            blocks = []
            
            # Восстанавливаем блоки из метаданных
            for block_meta in video_info["blocks"]:
                # Фильтруем субтитры для этого блока
                block_subtitles = [
                    s for s in app_state.subtitles 
                    if s["start"] >= block_meta["start_time"] and s["start"] <= block_meta["end_time"]
                ]
                
                block = {
                    "start_time": block_meta["start_time"],
                    "end_time": block_meta["end_time"],
                    "title": block_meta["title"],
                    "subtitles": block_subtitles,
                    "content_text": " ".join([s["text"] for s in block_subtitles])
                }
                
                blocks.append(block)
            
            app_state.subtitle_blocks = blocks
            
            # Генерируем оглавление
            app_state.table_of_contents = generate_table_of_contents(blocks)
        
        # Создаем QA цепочку
        app_state.qa_chain = setup_qa_chain(app_state.vectordb, "huggingface")
        
        # Создаем форматированный вывод
        video_info_html = f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div>
                <h3>{video_info.get('title', 'Неизвестное название')}</h3>
                <p>ID: {video_id}</p>
                <p>Язык: {video_info.get('language', 'Неизвестен')}</p>
                <p>Создано: {video_info.get('created_at', 'Неизвестно')}</p>
            </div>
        </div>
        """
        
        # Форматируем субтитры
        subtitles_markdown = format_subtitles(app_state.subtitles)
        
        return (
            f"✅ База данных успешно загружена для видео ID: {video_id}",
            video_info_html,
            subtitles_markdown
        )
        
    except Exception as e:
        logger.error(f"Ошибка загрузки базы данных: {e}")
        return (
            f"❌ Ошибка: {str(e)}",
            "",
            ""
        )


# ==============================================
# 9. GRADIO INTERFACE
# ==============================================
def create_gradio_interface():
    """Create Gradio interface with database loading functionality
    
    Returns:
        Gradio interface
    """
    # CSS для стилизации
    css = """
    .container {max-width: 1000px; margin: auto;}
    .title {text-align: center; margin-bottom: 20px;}
    .subtitle {text-align: center; margin-bottom: 30px; color: #666;}
    .tab-content {padding: 20px; border: 1px solid #ddd; border-radius: 5px;}
    .database-item {margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; cursor: pointer;}
    .database-item:hover {background-color: #f5f5f5;}
    .block-item {padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px; cursor: pointer;}
    .block-item:hover {background-color: #f0f8ff;}
    """
    
    with gr.Blocks(css=css) as demo:
        gr.HTML("""
        <div class="title">
            <h1>Обработчик субтитров YouTube</h1>
        </div>
        <div class="subtitle">
            <p>Извлечение, анализ и чат с субтитрами YouTube видео</p>
        </div>
        """)
        
        # Общие элементы для обоих режимов
        status_html = gr.HTML(label="Статус")
        video_info_html = gr.HTML(label="Информация о видео")
        
        # Используем Tabs вместо TabItem для создания простой структуры вкладок
        with gr.Tabs() as input_tabs:
            # Вкладка для обработки YouTube видео
            with gr.Tab(label="Обработка YouTube видео"):
                with gr.Row():
                    with gr.Column(scale=3):
                        youtube_url = gr.Textbox(
                            label="Ссылка на YouTube видео", 
                            placeholder="Введите ссылку на YouTube видео",
                            interactive=True
                        )
                        language_dropdown = gr.Dropdown(
                            choices=[
                                {"value": "en", "text": "English"},
                                {"value": "ru", "text": "Russian"},
                                {"value": "es", "text": "Spanish"},
                                {"value": "fr", "text": "French"},
                                {"value": "de", "text": "German"}
                            ],
                            value="en",
                            label="Предпочитаемый язык субтитров",
                            interactive=True
                        )
                        embedding_dropdown = gr.Dropdown(
                            choices=[
                                {"value": "huggingface", "text": "HuggingFace Embeddings"},
                                {"value": "openai", "text": "OpenAI Embeddings"}
                            ],
                            value="huggingface",
                            label="Модель эмбеддингов",
                            interactive=True
                        )
                    
                    with gr.Column(scale=1):
                        process_btn = gr.Button("Обработать видео", variant="primary")
            
            # Вкладка для загрузки существующей базы данных
            with gr.Tab(label="Загрузить базу данных"):
                with gr.Row():
                    with gr.Column():
                        # Кнопка обновления списка баз данных
                        refresh_db_btn = gr.Button("Обновить список баз данных")
                        
                        # Список доступных баз данных
                        available_dbs = get_saved_databases()
                        db_dropdown = gr.Dropdown(
                            choices=[{"value": db["video_id"], "text": f"{db['title']} ({db['video_id']})"}
                                    for db in available_dbs] if available_dbs else [],
                            label="Выберите базу данных",
                            interactive=True
                        )
                        
                        # Кнопка загрузки выбранной базы данных
                        load_db_btn = gr.Button("Загрузить выбранную базу данных", variant="primary")
        
        # Вкладки для вывода информации
        with gr.Tabs() as tabs:
            # Вкладка субтитров
            with gr.Tab(label="Субтитры"):
                subtitles_markdown = gr.Markdown()
            
            # Вкладка оглавления и блоков
            with gr.Tab(label="Содержание"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Оглавление видео")
                        table_of_contents = gr.Markdown()
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Информация о выбранном разделе")
                        with gr.Row():
                            block_dropdown = gr.Dropdown(
                                label="Выберите раздел для просмотра",
                                interactive=True
                            )
                            view_block_btn = gr.Button("Показать содержимое раздела", variant="primary")
                        
                        block_info = gr.Markdown()  # Для отображения информации о блоке
                        block_content = gr.Markdown()  # Для отображения полного содержимого блока
            
            # Вкладка перевода
            with gr.Tab(label="Перевод"):
                with gr.Row():
                    target_lang_dropdown = gr.Dropdown(
                        choices=[
                            {"value": "en", "text": "English"},
                            {"value": "ru", "text": "Russian"},
                            {"value": "es", "text": "Spanish"},
                            {"value": "fr", "text": "French"},
                            {"value": "de", "text": "German"},
                            {"value": "zh", "text": "Chinese"},
                            {"value": "ja", "text": "Japanese"},
                            {"value": "it", "text": "Italian"}
                        ],
                        value="en",
                        label="Перевести на",
                        interactive=True
                    )
                    translate_btn = gr.Button("Перевести субтитры")
                translated_output = gr.Markdown()
            
            # Вкладка чата
            with gr.Tab(label="Чат"):
                with gr.Row():
                    chat_model_dropdown = gr.Dropdown(
                        choices=[
                            {"value": "huggingface", "text": "HuggingFace Model"},
                            {"value": "openai", "text": "OpenAI Model"},
                            {"value": "groq", "text": "Groq Model"}
                        ],
                        value="huggingface",
                        label="Модель чата",
                        interactive=True
                    )
                
                chatbot = gr.Chatbot(height=400)
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Задайте вопрос о содержании видео...",
                        label="Ваш вопрос",
                        scale=4,
                        interactive=True
                    )
                    chat_btn = gr.Button("Отправить", scale=1)
                
                clear_btn = gr.Button("Очистить историю чата")
        
        # Обновленная функция process_video_and_update_toc с поддержкой глав
        def process_video_and_update_toc(url, embed_model, lang):
            """
            Обрабатывает видео и обновляет оглавление и список блоков
            
            Args:
                url: URL видео
                embed_model: Модель эмбеддингов
                lang: Язык субтитров
                
            Returns:
                Tuple with updated UI elements
            """
            # Если URL пустой, возвращаем сообщение об ошибке
            if not url:
                return (
                    "❌ Пожалуйста, введите ссылку на видео.",
                    "",
                    "",
                    "Оглавление не найдено",
                    gr.Dropdown.update(choices=[])
                )
            
            # Обрабатываем видео
            try:
                result = process_btn_wrapper(url, embed_model, lang)
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                return (
                    f"❌ Ошибка обработки видео: {str(e)}",
                    "",
                    "",
                    "Оглавление не найдено",
                    gr.Dropdown.update(choices=[])
                )
            
            # После обработки видео обновляем оглавление и список блоков
            toc = "Оглавление не найдено"
            try:
                if hasattr(app_state, 'table_of_contents') and app_state.table_of_contents:
                    toc = app_state.table_of_contents
                    
                    # Проверяем наличие глав YouTube
                    has_youtube_chapters = False
                    if hasattr(app_state, 'video_info'):
                        has_youtube_chapters = app_state.video_info.get("has_chapters", False)
                    
                    if has_youtube_chapters:
                        # Добавляем информацию о главах в статус
                        result = list(result)
                        result[0] = result[0] + " Используются главы YouTube."
                        result = tuple(result)
            except Exception as e:
                logger.error(f"Error accessing table of contents: {e}")
            
            # Обновляем выпадающий список блоков с более информативными названиями
            blocks_choices = []
            try:
                if hasattr(app_state, 'subtitle_blocks') and app_state.subtitle_blocks:
                    for i, block in enumerate(app_state.subtitle_blocks):
                        try:
                            start_time = format_time(block.get("start_time", 0))
                            title = block.get("title", f"Раздел {i+1}")
                            
                            # Сокращаем заголовок для выпадающего списка, если он слишком длинный
                            dropdown_title = title
                            if len(dropdown_title) > 40:
                                dropdown_title = dropdown_title[:37] + "..."
                            
                            # Добавляем значок для глав YouTube
                            chapter_icon = "🔖 " if block.get("is_youtube_chapter", False) else ""
                            
                            blocks_choices.append({
                                "value": str(i),
                                "text": f"{start_time} - {chapter_icon}{dropdown_title}"
                            })
                        except Exception as e:
                            logger.warning(f"Error formatting block for dropdown: {e}")
                            blocks_choices.append({
                                "value": str(i),
                                "text": f"Раздел {i+1} ({start_time})"
                            })
            except Exception as e:
                logger.error(f"Error generating blocks dropdown: {e}")
            
            # Обновляем оглавление и список блоков
            return result[0], result[1], result[2], toc, gr.Dropdown.update(choices=blocks_choices)


        # Обновление интерфейса для отображения информации о наличии глав
        def update_interface_for_chapters(demo):
            """
            Обновляет интерфейс для отображения информации о главах YouTube
            
            Args:
                demo: Gradio интерфейс
            """
            with demo:
                # Добавляем информацию о наличии глав в блок статуса
                chapters_info = gr.HTML(label="Информация о главах")
                
                # Функция для обновления информации о главах
                def update_chapters_info(url):
                    if not url:
                        return "Введите URL видео для проверки наличия глав"
                    
                    video_id = extract_video_id(url)
                    if not video_id:
                        return "❌ Некорректный URL видео"
                    
                    # Проверяем наличие глав
                    sources = check_chapter_sources(video_id)
                    
                    if sources["has_chapters"]:
                        sources_text = ", ".join(sources["sources"])
                        chapters_count = sources.get("chapters_count", 0) or sources.get("api_chapters_count", 0)
                        return f"✅ Найдены главы YouTube ({chapters_count}). Источник: {sources_text}"
                    else:
                        return "ℹ️ Главы YouTube не найдены. Будет использовано автоматическое разбиение."
                
                # Добавляем обработчик события для проверки наличия глав
                youtube_url.change(
                    fn=update_chapters_info,
                    inputs=[youtube_url],
                    outputs=[chapters_info]
                )
                
        # Функция для добавления дополнительной настройки для использования глав YouTube
        def add_youtube_chapters_option(demo):
            """
            Добавляет опцию для выбора использования глав YouTube
            
            Args:
                demo: Gradio интерфейс
            """
            with demo:
                # Добавляем чекбокс для выбора
                use_youtube_chapters = gr.Checkbox(
                    label="Использовать главы YouTube (если доступны)",
                    value=True,
                    interactive=True
                )
                
                # Обновляем app_state в соответствии с выбором
                def update_chapters_preference(use_chapters):
                    app_state.use_youtube_chapters = use_chapters
                    return f"{'✅' if use_chapters else '❌'} Использование глав YouTube: {'включено' if use_chapters else 'отключено'}"
                
                # Добавляем обработчик события
                use_youtube_chapters.change(
                    fn=update_chapters_preference,
                    inputs=[use_youtube_chapters],
                    outputs=[gr.HTML(label="Статус глав")]
                )

        
        # Установка обработчиков событий
        # Обработка YouTube видео
        process_btn.click(
            fn=process_video_and_update_toc,
            inputs=[youtube_url, embedding_dropdown, language_dropdown],
            outputs=[status_html, video_info_html, subtitles_markdown, table_of_contents, block_dropdown]
        )
        
        # Обновление списка баз данных
        def update_db_dropdown():
            databases = get_saved_databases()
            return gr.Dropdown.update(
                choices=[{"value": db["video_id"], "text": f"{db['title']} ({db['video_id']})"}
                         for db in databases] if databases else []
            )
        
        refresh_db_btn.click(
            fn=update_db_dropdown,
            inputs=[],
            outputs=[db_dropdown]
        )
        
        # Загрузка выбранной базы данных и обновление оглавления
        def load_selected_db_and_update_toc(selected_db_id):
            # Если ID не выбран, возвращаем предупреждение
            if not selected_db_id:
                return (
                    "⚠️ Пожалуйста, выберите базу данных из списка.",
                    "",
                    "",
                    "Оглавление не найдено",
                    gr.Dropdown.update(choices=[])
                )
            
            # Если получаем словарь из Dropdown, извлекаем значение
            if isinstance(selected_db_id, dict):
                selected_db_id = selected_db_id.get("value")
                if not selected_db_id:
                    return (
                        "⚠️ Некорректный выбор базы данных.",
                        "",
                        "",
                        "Оглавление не найдено",
                        gr.Dropdown.update(choices=[])
                    )
            
            # Загружаем базу данных
            try:
                status, info, subtitles = load_selected_db(selected_db_id)
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                return (
                    f"❌ Ошибка загрузки базы данных: {str(e)}",
                    "",
                    "",
                    "Оглавление не найдено",
                    gr.Dropdown.update(choices=[])
                )
            
            # Обновляем оглавление и список блоков
            toc = "Оглавление не найдено"
            blocks_choices = []
            
            try:
                if hasattr(app_state, 'subtitles') and app_state.subtitles:
                    # Разбиваем субтитры на блоки, если они еще не разбиты
                    if not hasattr(app_state, 'subtitle_blocks') or not app_state.subtitle_blocks:
                        try:
                            blocks, toc = process_subtitles_with_blocks(app_state.subtitles, app_state.video_info)
                        except Exception as e:
                            logger.error(f"Error processing subtitles into blocks: {e}")
                    else:
                        if hasattr(app_state, 'table_of_contents'):
                            toc = app_state.table_of_contents
                    
                    # Обновляем выпадающий список блоков с более информативными названиями
                    if hasattr(app_state, 'subtitle_blocks') and app_state.subtitle_blocks:
                        for i, block in enumerate(app_state.subtitle_blocks):
                            try:
                                start_time = format_time(block.get("start_time", 0))
                                title = block.get("title", f"Раздел {i+1}")
                                
                                # Сокращаем заголовок для выпадающего списка
                                dropdown_title = title
                                if len(dropdown_title) > 40:
                                    dropdown_title = dropdown_title[:37] + "..."
                                
                                blocks_choices.append({
                                    "value": str(i),
                                    "text": f"{start_time} - {dropdown_title}"
                                })
                            except Exception as e:
                                logger.warning(f"Error formatting block for dropdown: {e}")
                                blocks_choices.append({
                                    "value": str(i),
                                    "text": f"Раздел {i+1}"
                                })
            except Exception as e:
                logger.error(f"Error updating table of contents: {e}")
            
            return status, info, subtitles, toc, gr.Dropdown.update(choices=blocks_choices)
        
        load_db_btn.click(
            fn=load_selected_db_and_update_toc,
            inputs=[db_dropdown],
            outputs=[status_html, video_info_html, subtitles_markdown, table_of_contents, block_dropdown]
        )
        
        # Функция для обработки выбора блока из оглавления
        def handle_block_selection(block_index):
            """
            Обрабатывает выбор блока из выпадающего списка и отображает информацию о нем
            
            Args:
                block_index: Индекс выбранного блока
                
            Returns:
                Информация о выбранном блоке
            """
            # Отображаем краткую информацию о блоке
            return display_toc_entry(block_index)
        
        # Обновляем обработчики для выпадающего списка блоков
        block_dropdown.change(
            fn=handle_block_selection,
            inputs=[block_dropdown],
            outputs=[block_info]
        )
        
        # Функция для безопасного отображения содержимого блока
        def safe_display_block_content(block_index):
            """
            Безопасно отображает содержимое выбранного блока
            
            Args:
                block_index: Индекс блока (может быть словарем, строкой или None)
                
            Returns:
                Строка в формате Markdown с содержимым блока
            """
            # Проверяем, есть ли блоки субтитров
            if not hasattr(app_state, 'subtitle_blocks') or not app_state.subtitle_blocks:
                return "Блоки субтитров не найдены. Сначала обработайте видео."
            
            # Обработка значения из выпадающего списка Gradio
            if isinstance(block_index, dict):
                block_index = block_index.get("value")
            
            # Проверка на None или пустое значение
            if block_index is None or block_index == "":
                return "Пожалуйста, выберите блок из выпадающего списка."
            
            try:
                # Преобразуем в целое число
                block_index = int(block_index)
                
                # Проверяем, находится ли индекс в допустимом диапазоне
                if block_index < 0 or block_index >= len(app_state.subtitle_blocks):
                    return f"Ошибка: индекс блока {block_index} вне диапазона (0-{len(app_state.subtitle_blocks)-1})"
                
                # Получаем блок по индексу
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
            except (ValueError, TypeError) as e:
                # Ошибка преобразования в int или другая ошибка типа
                logger.error(f"Error in display_block_content: {e}")
                return f"Ошибка при обработке индекса блока: {str(e)}. Пожалуйста, выберите блок из списка."
            except Exception as e:
                # Другие ошибки
                logger.error(f"Unexpected error in display_block_content: {e}")
                return f"Неожиданная ошибка: {str(e)}"
        
        # Отображение содержимого выбранного блока
        view_block_btn.click(
            fn=safe_display_block_content,
            inputs=[block_dropdown],
            outputs=[block_content]
        )
        
        # Перевод субтитров
        translate_btn.click(
            fn=translate_subtitle_text,
            inputs=[target_lang_dropdown],
            outputs=[translated_output]
        )
        
        # Чат с субтитрами
        chat_btn.click(
            fn=chat_with_subtitles,
            inputs=[chat_input, chatbot, chat_model_dropdown],
            outputs=[chatbot]
        ).then(
            lambda: "", # Очистка ввода после отправки
            None,
            [chat_input]
        )
        
        clear_btn.click(
            fn=lambda: [],
            inputs=None,
            outputs=[chatbot]
        )
        
        # При загрузке интерфейса автоматически заполняем выпадающий список с базами данных
        demo.load(
            fn=update_db_dropdown,
            inputs=None,
            outputs=[db_dropdown]
        )
    
    return demo

# Функция-обертка для кнопки обработки видео
def process_btn_wrapper(url, embed_model, lang):
    if isinstance(embed_model, dict):
        embed_model = embed_model.get("value", "huggingface")
    
    if isinstance(lang, dict):
        lang = lang.get("value", "en")
    
    return process_video(url, embed_model, lang)