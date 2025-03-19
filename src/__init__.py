# src/__init__.py
class AppState:
    """Global application state"""
    def __init__(self):
        self.subtitles = []
        self.video_info = {}
        self.vectordb = None
        self.qa_chain = None
        self.chat_history = []
        self.current_model = "huggingface"
        self.subtitle_blocks = []
        self.table_of_contents = ""
        self.use_youtube_chapters = True

# Create global app state
app_state = AppState()