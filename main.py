# main.py
import os
import logging
import sys
from src.config import BASE_PATH, ENV_PATH, DB_PATH, LOGS_PATH, api_status, logger
from src.interface import create_gradio_interface

def main():
    """Основная функция для запуска приложения"""
    print("Запуск StudyPal...")
    
    # Выводим статус API
    print("\nСтатус API ключей:")
    for api, status in api_status.items():
        if status:
            print(f"✅ {api.capitalize()} API: настроен")
        else:
            print(f"❌ {api.capitalize()} API: не настроен")
    
    print(f"\nПути приложения:")
    print(f"База данных: {DB_PATH}")
    print(f"Файл логов: {LOGS_PATH}")
    
    print("\nСоздание интерфейса Gradio...")
    # Создаем и запускаем интерфейс Gradio
    demo = create_gradio_interface()
    
    # Выводим информацию о приложении
    print("""
    StudyPal: Обработчик субтитров YouTube
    -------------------------------------
    
    Это приложение позволяет:
    1. Извлекать субтитры из YouTube видео
    2. Обрабатывать и сохранять их в векторной базе данных
    3. Переводить субтитры на разные языки
    4. Вести чат с содержимым видео с помощью различных AI моделей
    5. Навигация по контенту с использованием глав видео или автоматически созданных разделов
    
    Чтобы начать, введите ссылку на YouTube и нажмите 'Обработать видео'.
    """)
    
    # Запускаем приложение
    demo.launch(share=False)

if __name__ == "__main__":
    main()