# src/config.py
import os
import logging
from dotenv import load_dotenv
from datetime import datetime

# Define paths
BASE_PATH = '/content/drive/My Drive/Colab Notebooks/Langchain'
ENV_PATH = f'{BASE_PATH}/.env'
DB_PATH = f'{BASE_PATH}/chroma_db'
LOGS_PATH = f'{BASE_PATH}/logs'

# Create directories if they don't exist
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Setup logging
log_file = f'{LOGS_PATH}/app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("studypal")

# Function to safely get and set API tokens
def setup_api_tokens():
    """Setup API tokens from .env file and validate their presence"""
    # Check if .env file exists, if not create a template
    if not os.path.exists(ENV_PATH):
        logger.info(f"Creating template .env file at {ENV_PATH}")
        with open(ENV_PATH, 'w') as f:
            f.write("""# API Keys
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
YOUTUBE_DATA_API_KEY=your_youtube_api_key_here
""")
    
    # Load environment variables
    load_dotenv(ENV_PATH)
    
    api_status = {"huggingface": False, "openai": False, "groq": False, "youtube": False}
    
    # HuggingFace Token
    huggingface_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if huggingface_token and huggingface_token != "your_huggingface_token_here":
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token
        api_status["huggingface"] = True
    
    # OpenAI Token
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key and openai_api_key != "your_openai_key_here":
        os.environ['OPENAI_API_KEY'] = openai_api_key
        api_status["openai"] = True
    
    # Groq Token
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key and groq_api_key != "your_groq_key_here":
        os.environ['GROQ_API_KEY'] = groq_api_key
        api_status["groq"] = True
    
    # YouTube Data API Token
    youtube_api_key = os.getenv('YOUTUBE_DATA_API_KEY')
    if youtube_api_key and youtube_api_key != "your_youtube_api_key_here":
        os.environ['YOUTUBE_DATA_API_KEY'] = youtube_api_key
        api_status["youtube"] = True
    
    return api_status

# App configuration
APP_SETTINGS = {
    "min_block_duration": 60,     # Minimum block duration in seconds
    "min_pause_threshold": 3,     # Minimum pause threshold for block separation
    "max_block_size": 25,         # Maximum number of subtitles in one block
    "chunk_size": 1000,           # Chunk size for vector database
    "chunk_overlap": 100,         # Chunk overlap
    "use_youtube_chapters": True, # Use YouTube chapters by default
    "default_language": "en",     # Default subtitle language
    "default_embedding": "huggingface", # Default embedding model
    "default_chat_model": "huggingface" # Default chat model
}

# Available languages for subtitles
AVAILABLE_LANGUAGES = [
    {"value": "en", "text": "English"},
    {"value": "ru", "text": "Russian"},
    {"value": "es", "text": "Spanish"},
    {"value": "fr", "text": "French"},
    {"value": "de", "text": "German"}
]

# Available models
CHAT_MODELS = [
    {"value": "huggingface", "text": "HuggingFace Model"},
    {"value": "openai", "text": "OpenAI Model"},
    {"value": "groq", "text": "Groq Model"}
]

EMBEDDING_MODELS = [
    {"value": "huggingface", "text": "HuggingFace Embeddings"},
    {"value": "openai", "text": "OpenAI Embeddings"}
]

# Translation languages
TRANSLATION_LANGUAGES = [
    {"value": "en", "text": "English"},
    {"value": "ru", "text": "Russian"},
    {"value": "es", "text": "Spanish"},
    {"value": "fr", "text": "French"},
    {"value": "de", "text": "German"},
    {"value": "zh", "text": "Chinese"},
    {"value": "ja", "text": "Japanese"},
    {"value": "it", "text": "Italian"}
]

# Initialize API tokens
api_status = setup_api_tokens()