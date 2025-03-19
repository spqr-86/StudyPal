import os
from google.colab import drive
from src.config import BASE_PATH, ENV_PATH, DB_PATH, LOGS_PATH, api_status, logger
from src.utils import display_info
from src.interface import create_gradio_interface

def main():
    """Main function to run the application"""
    # Mount Google Drive for persistent storage
    try:
        drive.mount('/content/drive')
        logger.info("Google Drive mounted successfully")
    except Exception as e:
        logger.warning(f"Failed to mount Google Drive: {e}")
    
    # Display API status
    for api, status in api_status.items():
        if status:
            display_info(f"{api.capitalize()} API", "✅ API token configured successfully", "success")
        else:
            display_info(f"{api.capitalize()} API", "⚠️ API token not found or not set", "warning")
    
    # Create and launch Gradio interface
    demo = create_gradio_interface()
    
    # Display app info
    display_info(
        "StudyPal", 
        """
        This application allows you to:
        1. Extract subtitles from YouTube videos
        2. Process and store them in a vector database
        3. Translate subtitles to different languages
        4. Chat with the video content using various AI models
        5. Navigate content using video chapters or auto-generated sections
        
        To get started, enter a YouTube URL and click 'Process Video'.
        """,
        "info"
    )
    
    # Launch the app
    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    main()