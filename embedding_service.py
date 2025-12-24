"""
Модуль для генерации эмбеддингов через Ollama
"""
import requests
from typing import List, Dict, Callable, Optional
import time
from logger import setup_logger

logger = setup_logger("embedding_service")
OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "nomic-embed-text"


def generate_embedding(text: str) -> List[float]:
    """
    Генерирует эмбеддинг для текста через Ollama
    
    Args:
        text: Текст для эмбеддинга
        
    Returns:
        Список чисел (вектор эмбеддинга)
    """
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }
    
    try:
        logger.info(f"🔄 Отправка запроса на генерацию эмбеддинга: {url} (длина текста: {len(text)} символов)")
        logger.debug(f"Payload: model={MODEL_NAME}, text_preview={text[:100]}...")
        
        response = requests.post(url, json=payload, timeout=60)
        logger.debug(f"Статус ответа: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding", [])
        
        if not embedding:
            logger.warning("⚠️ Получен пустой эмбеддинг!")
        else:
            logger.info(f"✅ Эмбеддинг успешно сгенерирован (размерность: {len(embedding)})")
        
        return embedding
    except requests.exceptions.Timeout as e:
        logger.error(f"⏱️ Таймаут при генерации эмбеддинга: {e}")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP ошибка при генерации эмбеддинга: {e}")
        logger.error(f"Ответ сервера: {response.text if 'response' in locals() else 'N/A'}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Ошибка при генерации эмбеддинга: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка при генерации эмбеддинга: {e}", exc_info=True)
        raise


def generate_embeddings_for_documents(
    documents: List[Dict], 
    batch_size: int = 1,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[Dict]:
    """
    Генерирует эмбеддинги для всех документов
    
    Args:
        documents: Список документов с текстом
        batch_size: Размер батча (пока не используется, т.к. Ollama принимает по одному)
        progress_callback: Функция обратного вызова для отслеживания прогресса
                          Принимает (current, total, document_name)
        
    Returns:
        Список документов с добавленными эмбеддингами
    """
    total = len(documents)
    documents_with_embeddings = []
    
    logger.info(f"🚀 Начало генерации эмбеддингов для {total} чанков")
    logger.info(f"📍 Ollama сервер: {OLLAMA_HOST}")
    logger.info(f"🤖 Модель: {MODEL_NAME}")
    logger.info(f"🔗 URL для эмбеддингов: {OLLAMA_HOST}/api/embeddings")
    
    # Проверяем доступность сервера
    try:
        test_response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if test_response.status_code == 200:
            logger.info("✅ Ollama сервер доступен")
        else:
            logger.warning(f"⚠️ Ollama сервер вернул статус {test_response.status_code}")
    except Exception as e:
        logger.error(f"❌ Не удалось подключиться к Ollama серверу: {e}")
        raise
    
    start_time = time.time()
    
    for idx, doc in enumerate(documents):
        document_name = doc.get('document', 'unknown')
        chunk_id = doc.get('chunk_id', 'unknown')
        current = idx + 1
        
        logger.info(f"[{current}/{total}] Обработка: {document_name} (чанк {chunk_id})")
        
        # Вызываем callback для обновления прогресса
        if progress_callback:
            try:
                progress_callback(current, total, document_name)
            except Exception as e:
                logger.warning(f"Ошибка в progress_callback: {e}")
        
        try:
            embedding = generate_embedding(doc['text'])
            doc['embedding'] = embedding
            documents_with_embeddings.append(doc)
            
            # Логируем прогресс каждые 10 чанков или на последнем
            if current % 10 == 0 or current == total:
                elapsed = time.time() - start_time
                rate = current / elapsed if elapsed > 0 else 0
                remaining = (total - current) / rate if rate > 0 else 0
                logger.info(
                    f"Прогресс: {current}/{total} ({current*100//total}%) | "
                    f"Скорость: {rate:.2f} чанков/сек | "
                    f"Осталось: {remaining:.1f} сек"
                )
            
            # Небольшая задержка, чтобы не перегружать сервер
            if idx < total - 1:
                time.sleep(0.1)
        except Exception as e:
            logger.error(
                f"Ошибка при обработке документа {document_name}, "
                f"чанк {chunk_id}: {e}"
            )
            # Пропускаем документ с ошибкой
    
    elapsed_total = time.time() - start_time
    logger.info(
        f"Генерация эмбеддингов завершена: {len(documents_with_embeddings)}/{total} "
        f"успешно обработано за {elapsed_total:.2f} секунд"
    )
    
    return documents_with_embeddings

