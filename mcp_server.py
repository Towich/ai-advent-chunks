"""
MCP-сервер для поиска по документам с использованием эмбеддингов
"""
import asyncio
import argparse
from typing import Dict, Any, Optional
from fastmcp import FastMCP
from embedding_service import generate_embedding
from index_manager import search_relevant_chunks_with_stats, load_threshold
from reranking_service import rerank_chunks
from logger import setup_logger

logger = setup_logger("mcp_server")

# Глобальная переменная для хранения флага реранкинга
ENABLE_RERANKING = False
# Глобальная переменная для хранения провайдера реранкинга
RERANK_PROVIDER = None  # "local" или "deepseek", если None - используется значение из reranking_service

# Создаем экземпляр FastMCP сервера
mcp = FastMCP("Document Search Server")


@mcp.tool()
async def search_documents(
    query: str,
    top_k: int = 5,
    min_similarity: Optional[float] = None
) -> Dict[str, Any]:
    """
    Ищет релевантные документы по текстовому запросу. Использует RAG-систему поиска
    
    Args:
        query: Текстовый запрос для поиска
        top_k: Количество наиболее релевантных результатов для возврата (по умолчанию 5)
        min_similarity: Минимальное значение косинусного сходства для фильтрации (0.0-1.0).
                       Если не указано, используется сохраненный порог.
    
    Returns:
        Словарь с результатами поиска, содержащий:
        - results: список найденных чанков с информацией о документе, тексте и сходстве
        - stats: статистика поиска (количество проверенных, отфильтрованных чанков и т.д.)
    """
    try:
        logger.info(f"Получен запрос на поиск: '{query}' (top_k={top_k}, min_similarity={min_similarity})")
        
        # Генерируем эмбеддинг для запроса
        logger.info("Генерация эмбеддинга для запроса")
        query_embedding = await asyncio.to_thread(generate_embedding, query)
        
        if not query_embedding:
            logger.error("Не удалось сгенерировать эмбеддинг для запроса")
            return {
                "error": "Не удалось сгенерировать эмбеддинг для запроса",
                "results": [],
                "stats": {}
            }
        
        # Используем сохраненный порог, если min_similarity не указан
        if min_similarity is None:
            min_similarity = await asyncio.to_thread(load_threshold)
            logger.info(f"Используется сохраненный порог: {min_similarity}")
        
        # Ищем релевантные чанки
        logger.info("Поиск релевантных чанков в индексе")
        
        # Если реранкинг включен, берем больше результатов для реранкинга
        if ENABLE_RERANKING:
            search_top_k = max(top_k * 2, 30)  # Берем минимум в 2 раза больше или 30
        else:
            search_top_k = top_k  # Берем только необходимое количество
        
        results, stats = await asyncio.to_thread(
            search_relevant_chunks_with_stats,
            query_embedding,
            top_k=search_top_k,
            min_similarity=min_similarity
        )
        
        # Применяем реранкинг через LLM только если он включен
        if ENABLE_RERANKING:
            logger.info(f"Применение реранкинга для {len(results)} найденных чанков (провайдер: {RERANK_PROVIDER or 'default'})")
            reranked_results = await asyncio.to_thread(
                rerank_chunks,
                query,
                results,
                max_chunks=top_k,
                provider=RERANK_PROVIDER
            )
            logger.info(f"После реранкинга осталось {len(reranked_results)} релевантных чанков")
        else:
            logger.info(f"Реранкинг отключен, возвращаем {len(results)} найденных чанков")
            # Ограничиваем результаты до top_k, если их больше
            reranked_results = results[:top_k]
        
        # Логируем найденные чанки
        logger.info(f"Найдено {len(reranked_results)} релевантных чанков:")
        for idx, (chunk, similarity) in enumerate(reranked_results, 1):
            document_name = chunk.get('document', 'unknown')
            chunk_id = chunk.get('chunk_id', 'unknown')
            chunk_text = chunk.get('text', '')
            # Обрезаем текст для лога, если он слишком длинный
            chunk_preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            logger.info(f"  [{idx}] Документ: {document_name}, Чанк ID: {chunk_id}, Сходство: {similarity:.4f}")
            logger.info(f"      Контент: {chunk_preview}")
        
        # Форматируем результаты для возврата
        formatted_results = []
        for chunk, similarity in reranked_results:
            formatted_results.append({
                "document": chunk.get("document", "unknown"),
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "text": chunk.get("text", ""),
                "similarity": round(similarity, 4)
            })
        
        response = {
            "query": query,
            "results": formatted_results,
            "stats": {
                "total_checked": stats.get("total_checked", 0),
                "total_filtered": stats.get("total_filtered", 0),
                "total_rejected": stats.get("total_rejected", 0),
                "min_similarity": stats.get("min_similarity", min_similarity),
                "best_similarity": round(stats.get("best_similarity", 0.0), 4),
                "best_filtered_similarity": round(stats.get("best_filtered_similarity", 0.0), 4),
                "results_count": len(formatted_results),
                "reranked": ENABLE_RERANKING,
                "before_reranking_count": len(results),
                "after_reranking_count": len(reranked_results)
            }
        }
        
        logger.info(f"Найдено {len(formatted_results)} результатов для запроса '{query}'")
        return response
        
    except Exception as e:
        logger.error(f"Ошибка при поиске документов: {e}", exc_info=True)
        return {
            "error": f"Ошибка при поиске: {str(e)}",
            "results": [],
            "stats": {}
        }


# Точка входа для запуска сервера
if __name__ == "__main__":
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="MCP-сервер для поиска по документам")
    parser.add_argument(
        "--enable-reranking",
        action="store_true",
        help="Включить RAG reranking (по умолчанию выключен)"
    )
    parser.add_argument(
        "--rerank-provider",
        type=str,
        choices=["local", "deepseek"],
        default=None,
        help="Провайдер для реранкинга: 'local' (локальная LLM) или 'deepseek' (DeepSeek через Hugging Face). "
             "Если не указан, используется значение из переменной окружения RERANK_PROVIDER или 'local'"
    )
    args, unknown = parser.parse_known_args()
    
    # Устанавливаем глобальный флаг реранкинга
    ENABLE_RERANKING = args.enable_reranking
    
    # Устанавливаем провайдера реранкинга
    if args.rerank_provider:
        RERANK_PROVIDER = args.rerank_provider
    else:
        import os
        RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", None)
    
    if ENABLE_RERANKING:
        provider_info = f" (провайдер: {RERANK_PROVIDER or 'default'})"
        logger.info(f"RAG reranking включен{provider_info}")
    else:
        logger.info("RAG reranking выключен")
    
    # FastMCP автоматически обрабатывает аргументы командной строки
    # для работы через stdio или другие транспорты
    # Передаем неизвестные аргументы в FastMCP
    mcp.run(transport="streamable-http")

