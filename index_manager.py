"""
Модуль для работы с индексом документов (сохранение/загрузка JSON)
"""
import json
import os
import math
from typing import List, Dict, Optional, Tuple
from logger import setup_logger

logger = setup_logger("index_manager")
INDEX_FILE = "document_index.json"
THRESHOLD_FILE = "search_threshold.json"
DEFAULT_THRESHOLD = 0.5  # Порог по умолчанию


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


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Вычисляет косинусное сходство между двумя векторами
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор
        
    Returns:
        Косинусное сходство (от -1 до 1)
    """
    if len(vec1) != len(vec2):
        logger.warning(f"Размерности векторов не совпадают: {len(vec1)} != {len(vec2)}")
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def save_threshold(threshold: float, threshold_path: str = THRESHOLD_FILE) -> None:
    """
    Сохраняет порог фильтрации в файл
    
    Args:
        threshold: Значение порога (от 0.0 до 1.0)
        threshold_path: Путь к файлу с порогом
    """
    try:
        with open(threshold_path, 'w', encoding='utf-8') as f:
            json.dump({"threshold": threshold}, f, indent=2)
        logger.info(f"Порог фильтрации сохранен: {threshold}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении порога: {e}", exc_info=True)


def load_threshold(threshold_path: str = THRESHOLD_FILE) -> float:
    """
    Загружает порог фильтрации из файла
    
    Args:
        threshold_path: Путь к файлу с порогом
        
    Returns:
        Значение порога или значение по умолчанию
    """
    if not os.path.exists(threshold_path):
        logger.info(f"Файл с порогом не найден, используется значение по умолчанию: {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD
    
    try:
        with open(threshold_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            threshold = data.get("threshold", DEFAULT_THRESHOLD)
            logger.info(f"Порог фильтрации загружен: {threshold}")
            return threshold
    except Exception as e:
        logger.warning(f"Ошибка при загрузке порога, используется значение по умолчанию: {e}")
        return DEFAULT_THRESHOLD


def search_relevant_chunks(
    query_embedding: List[float],
    index_path: str = INDEX_FILE,
    top_k: int = 5,
    min_similarity: Optional[float] = None
) -> List[Tuple[Dict, float]]:
    """
    Ищет наиболее релевантные чанки по запросу
    
    Args:
        query_embedding: Эмбеддинг запроса
        index_path: Путь к файлу индекса
        top_k: Количество наиболее релевантных чанков для возврата
        min_similarity: Минимальное значение косинусного сходства (если None, используется сохраненный порог)
        
    Returns:
        Список кортежей (чанк, similarity_score), отсортированный по убыванию сходства
    """
    results, _ = search_relevant_chunks_with_stats(
        query_embedding, index_path, top_k, min_similarity
    )
    return results


def search_relevant_chunks_with_stats(
    query_embedding: List[float],
    index_path: str = INDEX_FILE,
    top_k: int = 5,
    min_similarity: Optional[float] = None
) -> Tuple[List[Tuple[Dict, float]], Dict]:
    """
    Ищет наиболее релевантные чанки по запросу и возвращает статистику
    
    Args:
        query_embedding: Эмбеддинг запроса
        index_path: Путь к файлу индекса
        top_k: Количество наиболее релевантных чанков для возврата
        min_similarity: Минимальное значение косинусного сходства (если None, используется сохраненный порог)
        
    Returns:
        Кортеж (список результатов, словарь со статистикой)
    """
    # Используем сохраненный порог, если min_similarity не указан
    if min_similarity is None:
        min_similarity = load_threshold()
    
    logger.info(f"Поиск релевантных чанков (top_k={top_k}, min_similarity={min_similarity})")
    
    documents = load_index(index_path)
    if documents is None:
        logger.warning("Индекс не найден для поиска")
        return [], {}
    
    if not documents:
        logger.warning("Индекс пуст")
        return [], {}
    
    # Вычисляем сходство для каждого чанка
    all_similarities = []
    filtered_similarities = []
    
    for doc in documents:
        if 'embedding' not in doc:
            logger.warning(f"Чанк {doc.get('chunk_id', 'unknown')} не имеет эмбеддинга")
            continue
        
        embedding = doc['embedding']
        similarity = cosine_similarity(query_embedding, embedding)
        all_similarities.append((doc, similarity))
        
        if similarity >= min_similarity:
            filtered_similarities.append((doc, similarity))
    
    # Сортируем по убыванию сходства
    all_similarities.sort(key=lambda x: x[1], reverse=True)
    filtered_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ-K отфильтрованных результатов
    top_results = filtered_similarities[:top_k]
    
    # Статистика
    total_checked = len(all_similarities)
    total_filtered = len(filtered_similarities)
    total_rejected = total_checked - total_filtered
    
    stats = {
        "total_checked": total_checked,
        "total_filtered": total_filtered,
        "total_rejected": total_rejected,
        "min_similarity": min_similarity,
        "best_similarity": all_similarities[0][1] if all_similarities else 0.0,
        "best_filtered_similarity": top_results[0][1] if top_results else 0.0,
        "all_top_results": all_similarities[:top_k] if all_similarities else []
    }
    
    logger.info(
        f"Найдено {len(top_results)} релевантных чанков из {total_checked} "
        f"(отфильтровано: {total_rejected}, прошло фильтр: {total_filtered})"
    )
    if top_results:
        logger.info(f"Лучшее сходство (с фильтром): {top_results[0][1]:.4f}")
    if all_similarities:
        logger.info(f"Максимальное сходство (без фильтра): {all_similarities[0][1]:.4f}")
    
    return top_results, stats

