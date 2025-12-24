"""
Модуль для работы с индексом документов (сохранение/загрузка JSON)
"""
import json
import os
from typing import List, Dict, Optional
from logger import setup_logger

logger = setup_logger("index_manager")
INDEX_FILE = "document_index.json"


def save_index(documents: List[Dict], index_path: str = INDEX_FILE) -> None:
    """
    Сохраняет индекс документов в JSON файл
    
    Args:
        documents: Список документов с эмбеддингами
        index_path: Путь к файлу индекса
    """
    logger.info(f"Начало сохранения индекса: {len(documents)} документов")
    
    # Конвертируем numpy массивы в списки, если они есть
    serializable_docs = []
    for doc in documents:
        serializable_doc = doc.copy()
        if 'embedding' in serializable_doc:
            # Убеждаемся, что embedding - это список
            if hasattr(serializable_doc['embedding'], 'tolist'):
                serializable_doc['embedding'] = serializable_doc['embedding'].tolist()
        serializable_docs.append(serializable_doc)
    
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        
        # Получаем размер файла
        file_size = os.path.getsize(index_path)
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Индекс сохранен в {index_path} ({len(documents)} документов, {file_size_mb:.2f} MB)")
        print(f"Индекс сохранен в {index_path} ({len(documents)} документов)")
    except Exception as e:
        logger.error(f"Ошибка при сохранении индекса: {e}", exc_info=True)
        raise


def load_index(index_path: str = INDEX_FILE) -> Optional[List[Dict]]:
    """
    Загружает индекс документов из JSON файла
    
    Args:
        index_path: Путь к файлу индекса
        
    Returns:
        Список документов или None, если файл не найден
    """
    if not os.path.exists(index_path):
        logger.warning(f"Индекс не найден: {index_path}")
        return None
    
    try:
        logger.info(f"Загрузка индекса из {index_path}")
        with open(index_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        file_size = os.path.getsize(index_path)
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Индекс загружен из {index_path} ({len(documents)} документов, {file_size_mb:.2f} MB)")
        print(f"Индекс загружен из {index_path} ({len(documents)} документов)")
        return documents
    except Exception as e:
        logger.error(f"Ошибка при загрузке индекса: {e}", exc_info=True)
        print(f"Ошибка при загрузке индекса: {e}")
        return None


def get_index_stats(index_path: str = INDEX_FILE) -> Dict:
    """
    Возвращает статистику по индексу
    
    Args:
        index_path: Путь к файлу индекса
        
    Returns:
        Словарь со статистикой
    """
    logger.debug(f"Получение статистики по индексу: {index_path}")
    documents = load_index(index_path)
    if documents is None:
        logger.info("Статистика: индекс не найден")
        return {"status": "Индекс не найден"}
    
    # Подсчитываем уникальные документы
    unique_docs = set(doc.get('document', 'unknown') for doc in documents)
    
    stats = {
        "status": "Индекс загружен",
        "total_chunks": len(documents),
        "unique_documents": len(unique_docs),
        "documents": list(unique_docs)
    }
    
    logger.info(f"Статистика индекса: {stats['total_chunks']} чанков, {stats['unique_documents']} документов")
    return stats

