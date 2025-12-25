"""
MCP-сервер для поиска по документам с использованием эмбеддингов
"""
import asyncio
from typing import Dict, Any, Optional
from fastmcp import FastMCP
from embedding_service import generate_embedding
from index_manager import search_relevant_chunks_with_stats, load_threshold
from logger import setup_logger

logger = setup_logger("mcp_server")

# Создаем экземпляр FastMCP сервера
mcp = FastMCP("Document Search Server")


@mcp.tool()
async def search_documents(
    query: str,
    top_k: int = 5,
    min_similarity: Optional[float] = None
) -> Dict[str, Any]:
    """
    Ищет релевантные документы по текстовому запросу.
    
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
        results, stats = await asyncio.to_thread(
            search_relevant_chunks_with_stats,
            query_embedding,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        # Форматируем результаты для возврата
        formatted_results = []
        for chunk, similarity in results:
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
                "results_count": len(formatted_results)
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
    # FastMCP автоматически обрабатывает аргументы командной строки
    # для работы через stdio или другие транспорты
    mcp.run(transport="streamable-http")

