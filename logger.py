"""
Модуль для настройки логирования
"""
import logging
import os
from datetime import datetime


def setup_logger(name: str = "indexing", log_file: str = "indexing.log") -> logging.Logger:
    """
    Настраивает логгер с записью в файл и консоль
    
    Args:
        name: Имя логгера
        log_file: Имя файла для логов
        
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Удаляем существующие обработчики, если есть
    logger.handlers = []
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Обработчик для файла с немедленным выводом
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.flush()  # Принудительный flush
    logger.addHandler(file_handler)
    
    # Обработчик для консоли с немедленным выводом
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.flush()  # Принудительный flush
    logger.addHandler(console_handler)
    
    return logger

