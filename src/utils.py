from IPython.display import display, HTML


def display_info(title, message, type="info"):
    styles = {
        "info": "background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 10px;",
        "success": "background-color: #ddffdd; border-left: 6px solid #4CAF50; padding: 10px;",
        "warning": "background-color: #ffffcc; border-left: 6px solid #ffeb3b; padding: 10px;",
        "error": "background-color: #ffdddd; border-left: 6px solid #f44336; padding: 10px;"
    }
    display(HTML(f"""
    <div style="{styles[type]}">
        <strong>{title}</strong><br>
        {message}
    </div>
    """))


def format_time(seconds):
    """
    Форматирует время в секундах в формат HH:MM:SS
    
    Args:
        seconds: Время в секундах
        
    Returns:
        Строка в формате HH:MM:SS
    """
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"