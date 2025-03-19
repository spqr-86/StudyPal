import numpy as np
import logging
from typing import List, Dict
from tqdm.notebook import tqdm
from src.config import logger
from src import app_state


def get_available_translation_languages():
    """Get available translation language pairs
    
    Returns:
        List of available language pairs
    """
    return [
        {"code": "en", "name": "English"},
        {"code": "ru", "name": "Russian"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"}
    ]

def translate_subtitles(subtitles: List[Dict], source_lang: str, target_lang: str) -> List[Dict]:
    """Translate subtitles to target language
    
    Args:
        subtitles: List of subtitle dictionaries
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        List of dictionaries with original and translated text
    """
    # Ensure we have string values, not dictionaries
    if isinstance(source_lang, dict):
        source_lang = source_lang.get("value", "en")
    
    if isinstance(target_lang, dict):
        target_lang = target_lang.get("value", "en")
    
    if source_lang == target_lang:
        # No translation needed
        return [
            {**subtitle, "translated_text": subtitle.get("text", "")}
            for subtitle in subtitles
        ]
    
    try:
        from transformers import pipeline
        
        # Select appropriate model based on language pair
        if source_lang == "en" and target_lang == "ru":
            model_name = "Helsinki-NLP/opus-mt-en-ru"
        elif source_lang == "ru" and target_lang == "en":
            model_name = "Helsinki-NLP/opus-mt-ru-en"
        elif source_lang == "en" and target_lang == "es":
            model_name = "Helsinki-NLP/opus-mt-en-es"
        elif source_lang == "es" and target_lang == "en":
            model_name = "Helsinki-NLP/opus-mt-es-en"
        elif source_lang == "en" and target_lang == "fr":
            model_name = "Helsinki-NLP/opus-mt-en-fr"
        elif source_lang == "fr" and target_lang == "en":
            model_name = "Helsinki-NLP/opus-mt-fr-en"
        elif source_lang == "en" and target_lang == "de":
            model_name = "Helsinki-NLP/opus-mt-en-de"
        elif source_lang == "de" and target_lang == "en":
            model_name = "Helsinki-NLP/opus-mt-de-en"
        elif source_lang == "en" and target_lang == "zh":
            model_name = "Helsinki-NLP/opus-mt-en-zh"
        elif source_lang == "zh" and target_lang == "en":
            model_name = "Helsinki-NLP/opus-mt-zh-en"
        else:
            # For other language pairs, try to use English as a pivot
            logger.info(f"Direct translation from {source_lang} to {target_lang} not available. Using English as pivot.")
            # First translate to English if source is not English
            if source_lang != "en":
                interim_model = f"Helsinki-NLP/opus-mt-{source_lang}-en"
                interim_translator = pipeline("translation", model=interim_model)
                
                interim_subtitles = []
                for subtitle in tqdm(subtitles, desc="Translating to English"):
                    text = subtitle.get("text", "")
                    try:
                        translation = interim_translator(text, max_length=512)
                        english_text = translation[0].get("translation_text", "")
                        interim_subtitles.append({
                            **subtitle,
                            "text": english_text
                        })
                    except Exception as e:
                        logger.warning(f"Error translating to English: {e}")
                        interim_subtitles.append(subtitle)
                
                # Now set up for English to target
                subtitles = interim_subtitles
                source_lang = "en"
            
            model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
        
        logger.info(f"Using translation model: {model_name}")
        translator = pipeline("translation", model=model_name)
        
        # Translate in batches to avoid memory issues
        batch_size = 8
        translated_subtitles = []
        
        for i in tqdm(range(0, len(subtitles), batch_size), desc="Translating subtitles"):
            batch = subtitles[i:i+batch_size]
            batch_texts = [subtitle.get("text", "") for subtitle in batch]
            
            try:
                # Translate batch
                translations = translator(batch_texts, max_length=512)
                
                # Combine original and translated text
                for j, translation in enumerate(translations):
                    subtitle = batch[j]
                    translated_text = translation.get("translation_text", "")
                    translated_subtitles.append({
                        **subtitle,
                        "translated_text": translated_text
                    })
            except Exception as e:
                logger.error(f"Batch translation error: {e}")
                # If batch fails, try one by one
                for subtitle in batch:
                    try:
                        text = subtitle.get("text", "")
                        translation = translator(text, max_length=512)
                        translated_text = translation[0].get("translation_text", "")
                        translated_subtitles.append({
                            **subtitle,
                            "translated_text": translated_text
                        })
                    except Exception as inner_e:
                        logger.error(f"Individual translation error: {inner_e}")
                        translated_subtitles.append({
                            **subtitle,
                            "translated_text": f"[Translation error]"
                        })
        
        return translated_subtitles
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        # Return original text if translation fails
        return [
            {**subtitle, "translated_text": f"[Translation error: {e}]"}
            for subtitle in subtitles
        ]

def translate_subtitle_text(target_language: str):
    """Translate subtitles to target language
    
    Args:
        target_language: Target language code
        
    Returns:
        Markdown formatted translated subtitles
    """
    if not hasattr(app_state, 'subtitles') or not app_state.subtitles:
        return "No subtitles loaded. Please process a video first."
    
    # Convert dropdown dictionary to string if needed
    if isinstance(target_language, dict):
        target_language = target_language.get("value", "en")
    
    source_lang = app_state.video_info.get("language_code", "en")
      
    try:
        translated = translate_subtitles(
            app_state.subtitles,
            source_lang,
            target_language
        )
        
        # Format translated subtitles
        result = ""
        for entry in translated:
            start_seconds = entry.get('start', 0)
            minutes, seconds = divmod(int(start_seconds), 60)
            hours, minutes = divmod(minutes, 60)
            
            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
            original = entry.get('text', '')
            translated = entry.get('translated_text', '')
            
            result += f"{timestamp}\n**Original:** {original}\n**Translated:** {translated}\n\n"
        
        return result
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"Error translating subtitles: {str(e)}"
