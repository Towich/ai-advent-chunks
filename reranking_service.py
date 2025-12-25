"""
Модуль для реранкинга результатов поиска через LLM
"""
import os
import requests
import re
import json
from typing import List, Dict, Tuple, Optional, Literal
from logger import setup_logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = setup_logger("reranking_service")

# Конфигурация для локальной LLM
LLM_HOST = os.getenv("LOCAL_LLM_HOST", "http://localhost:7999")
LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

# Конфигурация для DeepSeek через Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3.2:novita"
HF_BASE_URL = "https://router.huggingface.co/v1"

# Провайдер по умолчанию
DEFAULT_PROVIDER = os.getenv("RERANK_PROVIDER", "local")  # "local" или "deepseek"


def _rerank_with_local_llm(
    query: str,
    chunks: List[Tuple[Dict, float]],
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Выполняет реранкинг через локальную LLM
    
    Returns:
        Ответ от LLM с номерами релевантных чанков
    """
    url = f"{LLM_HOST}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 50
    }
    
    logger.info(f"Отправка запроса на реранкинг в локальную LLM: {url}")
    
    # Логируем тело запроса
    logger.info("=" * 60)
    logger.info("ЗАПРОС К ЛОКАЛЬНОЙ LLM (REQUEST BODY):")
    logger.info("=" * 60)
    try:
        log_payload = payload.copy()
        if 'messages' in log_payload:
            for msg in log_payload['messages']:
                content = msg.get('content', '')
                if len(content) > 1000:
                    msg['content'] = content[:1000] + f"\n... [обрезано, всего {len(content)} символов]"
        logger.info(json.dumps(log_payload, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning(f"Не удалось отформатировать payload для логирования: {e}")
        logger.info(f"Payload (raw): {payload}")
    logger.info("=" * 60)
    
    logger.info("Ожидание ответа от локальной LLM (может занять до 5 минут)...")
    
    response = requests.post(url, json=payload, timeout=360)
    response.raise_for_status()
    
    # Логируем тело ответа
    logger.info("=" * 60)
    logger.info("ОТВЕТ ОТ ЛОКАЛЬНОЙ LLM (RESPONSE BODY):")
    logger.info("=" * 60)
    
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.error(f"Ответ не является валидным JSON: {e}")
        logger.info(f"Статус код: {response.status_code}")
        logger.info(f"Тело ответа (raw): {response.text[:2000]}...")
        logger.info("=" * 60)
        raise
    
    try:
        logger.info(f"Статус код: {response.status_code}")
        logger.info(f"Заголовки ответа: {dict(response.headers)}")
        logger.info("Тело ответа (JSON):")
        logger.info(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning(f"Не удалось залогировать ответ: {e}")
        logger.info(f"Response data (raw): {data}")
    
    logger.info("=" * 60)
    
    if 'choices' in data and len(data['choices']) > 0:
        content = data['choices'][0]['message']['content'].strip()
        logger.info(f"Получен ответ от локальной LLM: {content}")
        return content
    else:
        logger.error("Неожиданная структура ответа от локальной LLM")
        raise ValueError("Неожиданная структура ответа от локальной LLM")


def _rerank_with_deepseek(
    query: str,
    chunks: List[Tuple[Dict, float]],
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Выполняет реранкинг через DeepSeek-V3.2 через Hugging Face API
    
    Returns:
        Ответ от LLM с номерами релевантных чанков
    """
    if OpenAI is None:
        raise ImportError("openai библиотека не установлена. Установите её: pip install openai")
    
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN не установлен. Установите переменную окружения HF_TOKEN")
    
    client = OpenAI(
        base_url=HF_BASE_URL,
        api_key=HF_TOKEN,
    )
    
    logger.info(f"Отправка запроса на реранкинг в DeepSeek через Hugging Face API")
    logger.info(f"Модель: {DEEPSEEK_MODEL}")
    
    # Логируем тело запроса
    logger.info("=" * 60)
    logger.info("ЗАПРОС К DEEPSEEK (REQUEST BODY):")
    logger.info("=" * 60)
    try:
        log_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        for msg in log_messages:
            content = msg.get('content', '')
            if len(content) > 1000:
                msg['content'] = content[:1000] + f"\n... [обрезано, всего {len(content)} символов]"
        logger.info(json.dumps({"model": DEEPSEEK_MODEL, "messages": log_messages}, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning(f"Не удалось отформатировать payload для логирования: {e}")
    logger.info("=" * 60)
    
    logger.info("Ожидание ответа от DeepSeek...")
    
    try:
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        content = completion.choices[0].message.content.strip()
        
        # Логируем ответ
        logger.info("=" * 60)
        logger.info("ОТВЕТ ОТ DEEPSEEK:")
        logger.info("=" * 60)
        logger.info(f"Получен ответ от DeepSeek: {content}")
        logger.info("=" * 60)
        
        return content
        
    except Exception as e:
        logger.error(f"Ошибка при запросе к DeepSeek: {e}", exc_info=True)
        raise


def rerank_chunks(
    query: str,
    chunks: List[Tuple[Dict, float]],
    max_chunks: Optional[int] = None,
    provider: Optional[Literal["local", "deepseek"]] = None
) -> List[Tuple[Dict, float]]:
    """
    Отправляет изначальный вопрос и найденные чанки в LLM для фильтрации релевантных.
    
    Args:
        query: Изначальный вопрос пользователя
        chunks: Список кортежей (чанк, similarity_score) - результаты поиска
        max_chunks: Максимальное количество чанков для возврата (если None, возвращает все отфильтрованные)
        provider: Провайдер для реранкинга ("local" или "deepseek"). Если None, используется DEFAULT_PROVIDER
    
    Returns:
        Список отфильтрованных чанков в том же формате (чанк, similarity_score)
    """
    if not chunks:
        logger.info("Нет чанков для реранкинга")
        return []
    
    # Определяем провайдера
    if provider is None:
        provider = DEFAULT_PROVIDER
    
    if provider not in ["local", "deepseek"]:
        logger.warning(f"Неизвестный провайдер: {provider}, используем 'local'")
        provider = "local"
    
    logger.info(f"Используется провайдер реранкинга: {provider}")
    
    # Логируем чанки до реранкинга
    logger.info(f"=" * 60)
    logger.info(f"РЕРАНКИНГ: Начало обработки {len(chunks)} чанков (провайдер: {provider})")
    logger.info(f"=" * 60)
    logger.info("Чанки ДО реранкинга:")
    for idx, (chunk, similarity) in enumerate(chunks, 1):
        document_name = chunk.get('document', 'unknown')
        chunk_id = chunk.get('chunk_id', 'unknown')
        chunk_text = chunk.get('text', '')
        logger.info(f"  [{idx}] Документ: {document_name}, Чанк ID: {chunk_id}, Сходство: {similarity:.4f}")
        logger.info(f"      Контент: {chunk_text}")
    
    try:
        # Формируем системный промпт с инструкциями
        system_prompt = """You are an assistant for filtering relevant documents.

Your task is to analyze document fragments (chunks) and determine which of them are actually relevant to the user's question.

IMPORTANT: return only the numbers of the relevant chunks, separated by commas, with no additional explanation. For example: "1, 3, 5"
If none of the chunks are relevant, return "0"."""
        
        # Формируем пользовательский промпт с вопросом и чанками
        chunks_text = ""
        for idx, (chunk, similarity) in enumerate(chunks, 1):
            chunk_text = chunk.get('text', '')
            document_name = chunk.get('document', 'unknown')
            chunks_text += f"\n--- Чанк {idx} (сходство: {similarity:.3f}, документ: {document_name}) ---\n{chunk_text}\n"
        
        user_prompt = f"""The user asked the following question:
"{query}"

Below are the found document fragments (chunks):

{chunks_text}

Analyze each chunk and identify which ones truly answer the user's question or contain relevant information."""
        
        logger.info(f"Количество чанков для фильтрации: {len(chunks)}")
        
        # Вызываем соответствующий провайдер
        if provider == "deepseek":
            content = _rerank_with_deepseek(query, chunks, system_prompt, user_prompt)
        else:  # provider == "local"
            content = _rerank_with_local_llm(query, chunks, system_prompt, user_prompt)
        
        # Парсим ответ - извлекаем номера чанков
        try:
            # Пробуем найти числа в ответе разными способами
            selected_indices = []
            
            # Ищем все числа в тексте
            numbers = re.findall(r'\d+', content)
            
            for num_str in numbers:
                try:
                    num = int(num_str)
                    if num == 0:
                        # Если LLM вернул 0, значит нет релевантных чанков
                        logger.info("=" * 60)
                        logger.info("РЕРАНКИНГ: LLM определил, что нет релевантных чанков")
                        logger.info(f"  До реранкинга: {len(chunks)} чанков")
                        logger.info(f"  После реранкинга: 0 чанков")
                        logger.info(f"  Отфильтровано: {len(chunks)} чанков")
                        logger.info("=" * 60)
                        return []
                    if 1 <= num <= len(chunks):
                        idx = num - 1  # Конвертируем в 0-based индекс
                        if idx not in selected_indices:  # Избегаем дубликатов
                            selected_indices.append(idx)
                except ValueError:
                    continue
            
            if not selected_indices:
                logger.warning(f"Не удалось извлечь номера чанков из ответа LLM: '{content}', возвращаем все чанки")
                logger.info("=" * 60)
                logger.info("РЕРАНКИНГ: Ошибка парсинга, возвращаем все чанки без изменений")
                logger.info(f"  До реранкинга: {len(chunks)} чанков")
                logger.info(f"  После реранкинга: {len(chunks)} чанков (без изменений)")
                logger.info("=" * 60)
                return chunks
            
            # Фильтруем чанки по выбранным индексам, сохраняя порядок из ответа LLM
            reranked_chunks = [chunks[i] for i in selected_indices if i < len(chunks)]
            
            # Ограничиваем количество, если указано
            if max_chunks is not None:
                reranked_chunks = reranked_chunks[:max_chunks]
            
            # Логируем чанки после реранкинга
            logger.info(f"=" * 60)
            logger.info(f"РЕРАНКИНГ: Завершен")
            logger.info(f"  До реранкинга: {len(chunks)} чанков")
            logger.info(f"  После реранкинга: {len(reranked_chunks)} чанков")
            logger.info(f"  Отфильтровано: {len(chunks) - len(reranked_chunks)} чанков")
            logger.info(f"=" * 60)
            logger.info("Чанки ПОСЛЕ реранкинга:")
            # selected_indices и reranked_chunks имеют одинаковый порядок
            for new_idx, original_idx in enumerate(selected_indices[:len(reranked_chunks)]):
                chunk, similarity = reranked_chunks[new_idx]
                document_name = chunk.get('document', 'unknown')
                chunk_id = chunk.get('chunk_id', 'unknown')
                chunk_text = chunk.get('text', '')
                original_position = original_idx + 1  # Конвертируем обратно в 1-based для отображения
                logger.info(f"  [{new_idx + 1}] (был #{original_position}) Документ: {document_name}, Чанк ID: {chunk_id}, Сходство: {similarity:.4f}")
                logger.info(f"      Контент: {chunk_text}")
            
            logger.info(f"=" * 60)
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге ответа LLM: {e}", exc_info=True)
            logger.error(f"Ответ LLM: {content}")
            logger.info("=" * 60)
            logger.info("РЕРАНКИНГ: Ошибка парсинга, возвращаем все чанки без изменений")
            logger.info(f"  До реранкинга: {len(chunks)} чанков")
            logger.info(f"  После реранкинга: {len(chunks)} чанков (без изменений)")
            logger.info("=" * 60)
            return chunks  # Возвращаем все чанки при ошибке парсинга
        
    except requests.exceptions.Timeout:
        logger.error("Таймаут при запросе к LLM для реранкинга")
        logger.info("=" * 60)
        logger.info("РЕРАНКИНГ: Таймаут, возвращаем все чанки без изменений")
        logger.info(f"  До реранкинга: {len(chunks)} чанков")
        logger.info(f"  После реранкинга: {len(chunks)} чанков (без изменений)")
        logger.info("=" * 60)
        return chunks  # Возвращаем все чанки при таймауте
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP ошибка при запросе к LLM для реранкинга: {e}")
        logger.info("=" * 60)
        logger.info("РЕРАНКИНГ: HTTP ошибка, возвращаем все чанки без изменений")
        logger.info(f"  До реранкинга: {len(chunks)} чанков")
        logger.info(f"  После реранкинга: {len(chunks)} чанков (без изменений)")
        logger.info("=" * 60)
        return chunks  # Возвращаем все чанки при ошибке
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к LLM для реранкинга: {e}", exc_info=True)
        logger.info("=" * 60)
        logger.info("РЕРАНКИНГ: Ошибка запроса, возвращаем все чанки без изменений")
        logger.info(f"  До реранкинга: {len(chunks)} чанков")
        logger.info(f"  После реранкинга: {len(chunks)} чанков (без изменений)")
        logger.info("=" * 60)
        return chunks  # Возвращаем все чанки при ошибке
    except (ImportError, ValueError) as e:
        logger.error(f"Ошибка конфигурации провайдера {provider}: {e}", exc_info=True)
        logger.info("=" * 60)
        logger.info("РЕРАНКИНГ: Ошибка конфигурации, возвращаем все чанки без изменений")
        logger.info(f"  До реранкинга: {len(chunks)} чанков")
        logger.info(f"  После реранкинга: {len(chunks)} чанков (без изменений)")
        logger.info("=" * 60)
        return chunks  # Возвращаем все чанки при ошибке конфигурации
    except Exception as e:
        logger.error(f"Неожиданная ошибка при реранкинге: {e}", exc_info=True)
        logger.info("=" * 60)
        logger.info("РЕРАНКИНГ: Неожиданная ошибка, возвращаем все чанки без изменений")
        logger.info(f"  До реранкинга: {len(chunks)} чанков")
        logger.info(f"  После реранкинга: {len(chunks)} чанков (без изменений)")
        logger.info("=" * 60)
        return chunks  # Возвращаем все чанки при неожиданной ошибке

