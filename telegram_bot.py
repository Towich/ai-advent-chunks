"""
Telegram бот для обработки документов и создания индекса с эмбеддингами
"""
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from document_processor import process_documents_from_folder
from embedding_service import generate_embeddings_for_documents, generate_embedding
from index_manager import (
    save_index, get_index_stats, search_relevant_chunks,
    search_relevant_chunks_with_stats, save_threshold, load_threshold
)
from logger import setup_logger

logger = setup_logger("telegram_bot")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    logger.info(f"Получена команда /start от пользователя {update.effective_user.id}")
    welcome_message = """
🤖 Бот для индексации документов с эмбеддингами

Доступные команды:
/start - Показать это сообщение
/index - Создать индекс из PDF и MD файлов в папке проекта
/stats - Показать статистику по индексу
/search <вопрос> - Найти релевантные чанки по вопросу (с фильтром)
/search_compare <вопрос> - Сравнить результаты с фильтром и без
/set_threshold <значение> - Установить порог фильтрации (0.0-1.0)
/get_threshold - Показать текущий порог фильтрации
/help - Показать справку
"""
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help"""
    help_text = """
📚 Справка по использованию бота:

/index - Запускает процесс индексации:
  1. Находит все PDF и MD файлы в папке проекта
  2. Извлекает текст из PDF или MD файлов
  3. Разбивает на чанки (для MD - умное разбиение по заголовкам)
  4. Генерирует эмбеддинги через Ollama (nomic-embed-text)
  5. Сохраняет индекс в document_index.json

/stats - Показывает статистику по существующему индексу

/search <вопрос> - Ищет релевантные чанки с фильтрацией:
  1. Генерирует эмбеддинг для вашего вопроса
  2. Ищет наиболее похожие чанки в индексе
  3. Фильтрует результаты по порогу сходства
  4. Возвращает топ-5 наиболее релевантных результатов

/search_compare <вопрос> - Сравнивает результаты с фильтром и без:
  Показывает разницу между поиском с фильтром и без него

/set_threshold <значение> - Устанавливает порог фильтрации (0.0-1.0):
  • 0.0-0.5 - низкий порог (больше результатов)
  • 0.5-0.7 - средний порог (сбалансированный)
  • 0.7-1.0 - высокий порог (только очень релевантные)
  Пример: /set_threshold 0.6

/get_threshold - Показывает текущий порог фильтрации

⚠️ Убедитесь, что:
  - Ollama запущен на http://127.0.0.1:11434
  - Модель nomic-embed-text установлена в Ollama
  - PDF и MD файлы находятся в папке проекта
"""
    await update.message.reply_text(help_text)


async def index_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /index - создание индекса"""
    chat_id = update.effective_chat.id
    logger.info(f"Получена команда /index от пользователя {update.effective_user.id}")
    
    await update.message.reply_text("🔄 Начинаю обработку документов...")
    
    try:
        # Обработка документов (в отдельном потоке, чтобы не блокировать)
        await update.message.reply_text("📄 Ищу PDF и MD файлы и извлекаю текст...")
        logger.info("Начало обработки PDF и MD файлов")
        # Обрабатываем по 5 страниц за раз для экономии памяти
        documents = await asyncio.to_thread(process_documents_from_folder)
        
        if not documents:
            logger.warning("PDF и MD файлы не найдены в папке проекта")
            await update.message.reply_text("❌ PDF и MD файлы не найдены в папке проекта")
            return
        
        unique_docs = len(set(d['document'] for d in documents))
        logger.info(f"Найдено {len(documents)} чанков из {unique_docs} документов")
        
        await update.message.reply_text(
            f"✅ Найдено {len(documents)} чанков из {unique_docs} документов\n"
            f"🔄 Начинаю генерацию эмбеддингов через Ollama..."
        )
        
        # Словарь для отслеживания прогресса (используется из синхронного потока)
        progress_state = {
            'current': 0,
            'total': len(documents),
            'document_name': '',
            'updated': False
        }
        
        # Сообщение для отслеживания прогресса
        progress_message = None
        
        # Callback для обновления прогресса (вызывается из синхронного потока)
        def progress_callback(current: int, total: int, document_name: str):
            """Сохраняет прогресс в общий словарь"""
            progress_state['current'] = current
            progress_state['total'] = total
            progress_state['document_name'] = document_name
            progress_state['updated'] = True
        
        # Асинхронная задача для обновления прогресса в Telegram
        async def update_progress_task():
            """Периодически обновляет сообщение о прогрессе"""
            nonlocal progress_message
            
            while True:
                await asyncio.sleep(2)  # Проверяем каждые 2 секунды
                
                if progress_state['updated']:
                    current = progress_state['current']
                    total = progress_state['total']
                    document_name = progress_state['document_name']
                    
                    if current > 0 and total > 0:
                        percentage = current * 100 // total
                        progress_bar_length = 20
                        filled = int(progress_bar_length * current / total)
                        bar = "█" * filled + "░" * (progress_bar_length - filled)
                        
                        progress_text = (
                            f"🔄 Генерация эмбеддингов\n\n"
                            f"📊 Прогресс: {current}/{total} ({percentage}%)\n"
                            f"{bar}\n\n"
                            f"📄 Текущий документ: {document_name}"
                        )
                        
                        try:
                            if progress_message is None:
                                # Создаем новое сообщение
                                progress_message = await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=progress_text
                                )
                            else:
                                # Обновляем существующее сообщение
                                await context.bot.edit_message_text(
                                    chat_id=chat_id,
                                    message_id=progress_message.message_id,
                                    text=progress_text
                                )
                        except Exception as e:
                            logger.warning(f"Не удалось обновить прогресс в Telegram: {e}")
                    
                    progress_state['updated'] = False
                
                # Проверяем, завершена ли обработка (когда current == total и не обновляется)
                if progress_state['current'] >= progress_state['total']:
                    await asyncio.sleep(1)  # Даем время на финальное обновление
                    if not progress_state['updated']:
                        break
        
        # Генерация эмбеддингов (в отдельном потоке, чтобы не блокировать event loop)
        async def process_embeddings():
            try:
                logger.info("=" * 60)
                logger.info("🚀 НАЧАЛО ГЕНЕРАЦИИ ЭМБЕДДИНГОВ")
                logger.info(f"📊 Количество документов для обработки: {len(documents)}")
                logger.info("=" * 60)
                
                # Запускаем задачу обновления прогресса
                progress_task = asyncio.create_task(update_progress_task())
                
                # Запускаем синхронную функцию в отдельном потоке
                logger.info("🔄 Вызов функции generate_embeddings_for_documents...")
                documents_with_embeddings = await asyncio.to_thread(
                    generate_embeddings_for_documents, 
                    documents,
                    1,  # batch_size
                    progress_callback
                )
                logger.info(f"✅ Функция generate_embeddings_for_documents завершена, получено {len(documents_with_embeddings)} документов")
                
                # Ждем завершения задачи обновления прогресса
                await progress_task
                
                if not documents_with_embeddings:
                    logger.error("Ошибка: не удалось сгенерировать эмбеддинги")
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="❌ Ошибка при генерации эмбеддингов"
                    )
                    return
                
                # Удаляем сообщение о прогрессе
                try:
                    if progress_message:
                        await context.bot.delete_message(
                            chat_id=chat_id,
                            message_id=progress_message.message_id
                        )
                except Exception as e:
                    logger.debug(f"Не удалось удалить сообщение о прогрессе: {e}")
                
                # Сохранение индекса (тоже в отдельном потоке)
                logger.info("Сохранение индекса в файл")
                await asyncio.to_thread(save_index, documents_with_embeddings)
                
                logger.info(f"Индекс успешно создан: {len(documents_with_embeddings)} чанков")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"✅ Индекс успешно создан!\n"
                         f"📊 Обработано: {len(documents_with_embeddings)} чанков\n"
                         f"💾 Сохранено в: document_index.json"
                )
            except Exception as e:
                logger.error(f"Ошибка при обработке: {e}", exc_info=True)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ Ошибка при обработке: {str(e)}"
                )
        
        # Запускаем обработку в фоне
        asyncio.create_task(process_embeddings())
        
    except Exception as e:
        logger.error(f"Ошибка в команде /index: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /stats - статистика по индексу"""
    try:
        stats = get_index_stats()
        
        if stats.get("status") == "Индекс не найден":
            await update.message.reply_text("❌ Индекс не найден. Используйте /index для создания индекса.")
            return
        
        stats_message = f"""
📊 Статистика индекса:

✅ Статус: {stats['status']}
📄 Всего чанков: {stats['total_chunks']}
📚 Уникальных документов: {stats['unique_documents']}

Документы:
"""
        for doc in stats['documents']:
            stats_message += f"  • {doc}\n"
        
        await update.message.reply_text(stats_message)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


def format_search_results(
    query: str,
    results: list,
    stats: dict = None,
    show_filter_info: bool = False
) -> list:
    """
    Форматирует результаты поиска для отправки в Telegram
    
    Args:
        query: Текст запроса
        results: Список результатов (чанк, similarity)
        stats: Словарь со статистикой (опционально)
        show_filter_info: Показывать ли информацию о фильтрации
        
    Returns:
        Список сообщений для отправки
    """
    TELEGRAM_MAX_LENGTH = 4096
    HEADER_LENGTH = 300
    MAX_CHUNK_LENGTH = TELEGRAM_MAX_LENGTH - HEADER_LENGTH
    
    messages = []
    
    # Формируем заголовок
    header = f"🔍 Результаты поиска по запросу: \"{query}\"\n\n"
    
    if show_filter_info and stats:
        threshold = stats.get('min_similarity', 0.0)
        total_checked = stats.get('total_checked', 0)
        total_filtered = stats.get('total_filtered', 0)
        total_rejected = stats.get('total_rejected', 0)
        best_similarity = stats.get('best_filtered_similarity', 0.0)
        
        header += (
            f"📊 Статистика фильтрации:\n"
            f"  • Порог отсечения: {threshold:.3f}\n"
            f"  • Проверено чанков: {total_checked}\n"
            f"  • Прошло фильтр: {total_filtered}\n"
            f"  • Отфильтровано: {total_rejected}\n"
            f"  • Лучшее сходство: {best_similarity:.3f}\n\n"
        )
    
    if not results:
        messages.append(header + "❌ Релевантные чанки не найдены.")
        return messages
    
    current_message = header
    
    for idx, (chunk, similarity) in enumerate(results, 1):
        chunk_text = chunk.get('text', '')
        document_name = chunk.get('document', 'unknown')
        chunk_id = chunk.get('chunk_id', 'unknown')
        
        # Обрезаем текст чанка, если он слишком длинный
        if len(chunk_text) > MAX_CHUNK_LENGTH:
            chunk_text = chunk_text[:MAX_CHUNK_LENGTH - 3] + "..."
        
        # Форматируем информацию о чанке
        chunk_info = (
            f"📄 Результат {idx} (сходство: {similarity:.3f})\n"
            f"📚 Документ: {document_name}\n"
            f"🔢 Чанк ID: {chunk_id}\n"
            f"📝 Текст:\n{chunk_text}\n\n"
            f"{'─' * 40}\n\n"
        )
        
        # Проверяем, поместится ли следующий чанк в текущее сообщение
        if len(current_message) + len(chunk_info) > TELEGRAM_MAX_LENGTH:
            # Сохраняем текущее сообщение и начинаем новое
            messages.append(current_message.rstrip())
            current_message = f"🔍 Результаты поиска (продолжение):\n\n{chunk_info}"
        else:
            current_message += chunk_info
    
    # Добавляем последнее сообщение
    if current_message:
        messages.append(current_message.rstrip())
    
    return messages


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /search - поиск релевантных чанков с фильтром"""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # Получаем вопрос из аргументов команды
    if not context.args:
        await update.message.reply_text(
            "❌ Пожалуйста, укажите вопрос для поиска.\n"
            "Пример: /search что такое чистый код?"
        )
        return
    
    query = " ".join(context.args)
    logger.info(f"Получен запрос на поиск от пользователя {user_id}: {query[:100]}")
    
    await update.message.reply_text("🔍 Ищу релевантные чанки...")
    
    try:
        # Генерируем эмбеддинг для вопроса
        logger.info("Генерация эмбеддинга для запроса")
        query_embedding = await asyncio.to_thread(generate_embedding, query)
        
        if not query_embedding:
            await update.message.reply_text("❌ Не удалось сгенерировать эмбеддинг для запроса")
            return
        
        # Ищем релевантные чанки с фильтром и статистикой
        logger.info("Поиск релевантных чанков в индексе")
        results, stats = await asyncio.to_thread(
            search_relevant_chunks_with_stats,
            query_embedding,
            top_k=5
        )
        
        if not results:
            threshold = stats.get('min_similarity', 0.0) if stats else 0.0
            best_similarity = stats.get('best_similarity', 0.0) if stats else 0.0
            await update.message.reply_text(
                f"❌ Релевантные чанки не найдены (порог: {threshold:.3f}).\n"
                f"Лучшее сходство без фильтра: {best_similarity:.3f}\n\n"
                f"💡 Попробуйте:\n"
                f"  • Уменьшить порог: /set_threshold {max(0.0, threshold - 0.1):.2f}\n"
                f"  • Посмотреть результаты без фильтра: /search_compare {query}"
            )
            return
        
        # Форматируем и отправляем результаты
        messages = format_search_results(query, results, stats, show_filter_info=True)
        
        for msg in messages:
            await context.bot.send_message(chat_id=chat_id, text=msg)
        
        logger.info(f"Отправлено {len(messages)} сообщений с результатами поиска")
        
    except Exception as e:
        logger.error(f"Ошибка в команде /search: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка при поиске: {str(e)}")


async def search_compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /search_compare - сравнение результатов с фильтром и без"""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # Получаем вопрос из аргументов команды
    if not context.args:
        await update.message.reply_text(
            "❌ Пожалуйста, укажите вопрос для поиска.\n"
            "Пример: /search_compare что такое чистый код?"
        )
        return
    
    query = " ".join(context.args)
    logger.info(f"Получен запрос на сравнение от пользователя {user_id}: {query[:100]}")
    
    await update.message.reply_text("🔍 Сравниваю результаты с фильтром и без...")
    
    try:
        # Генерируем эмбеддинг для вопроса
        logger.info("Генерация эмбеддинга для запроса")
        query_embedding = await asyncio.to_thread(generate_embedding, query)
        
        if not query_embedding:
            await update.message.reply_text("❌ Не удалось сгенерировать эмбеддинг для запроса")
            return
        
        # Поиск с фильтром
        results_filtered, stats_filtered = await asyncio.to_thread(
            search_relevant_chunks_with_stats,
            query_embedding,
            top_k=5
        )
        
        # Поиск без фильтра (min_similarity=0.0)
        results_no_filter, stats_no_filter = await asyncio.to_thread(
            search_relevant_chunks_with_stats,
            query_embedding,
            top_k=5,
            min_similarity=0.0
        )
        
        # Формируем сравнительное сообщение
        threshold = stats_filtered.get('min_similarity', 0.0)
        
        comparison_text = (
            f"🔍 Сравнение результатов поиска: \"{query}\"\n\n"
            f"📊 С ФИЛЬТРОМ (порог: {threshold:.3f}):\n"
            f"  • Найдено результатов: {len(results_filtered)}\n"
            f"  • Проверено чанков: {stats_filtered.get('total_checked', 0)}\n"
            f"  • Отфильтровано: {stats_filtered.get('total_rejected', 0)}\n"
            f"  • Лучшее сходство: {stats_filtered.get('best_filtered_similarity', 0.0):.3f}\n\n"
            f"📊 БЕЗ ФИЛЬТРА:\n"
            f"  • Найдено результатов: {len(results_no_filter)}\n"
            f"  • Проверено чанков: {stats_no_filter.get('total_checked', 0)}\n"
            f"  • Лучшее сходство: {stats_no_filter.get('best_similarity', 0.0):.3f}\n\n"
            f"{'=' * 40}\n\n"
        )
        
        # Добавляем результаты с фильтром
        if results_filtered:
            comparison_text += "✅ РЕЗУЛЬТАТЫ С ФИЛЬТРОМ:\n\n"
            for idx, (chunk, similarity) in enumerate(results_filtered[:3], 1):
                chunk_text = chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
                comparison_text += (
                    f"{idx}. Сходство: {similarity:.3f}\n"
                    f"   {chunk_text}\n\n"
                )
        else:
            comparison_text += "❌ С фильтром результатов не найдено\n\n"
        
        comparison_text += f"{'─' * 40}\n\n"
        
        # Добавляем результаты без фильтра
        if results_no_filter:
            comparison_text += "📋 РЕЗУЛЬТАТЫ БЕЗ ФИЛЬТРА (первые 3):\n\n"
            for idx, (chunk, similarity) in enumerate(results_no_filter[:3], 1):
                chunk_text = chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
                comparison_text += (
                    f"{idx}. Сходство: {similarity:.3f}\n"
                    f"   {chunk_text}\n\n"
                )
        
        await context.bot.send_message(chat_id=chat_id, text=comparison_text)
        
        # Отправляем полные результаты с фильтром отдельным сообщением
        if results_filtered:
            messages = format_search_results(query, results_filtered, stats_filtered, show_filter_info=True)
            for msg in messages:
                await context.bot.send_message(chat_id=chat_id, text=msg)
        
        logger.info(f"Отправлено сравнение результатов поиска")
        
    except Exception as e:
        logger.error(f"Ошибка в команде /search_compare: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка при сравнении: {str(e)}")


async def set_threshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /set_threshold - установка порога фильтрации"""
    if not context.args:
        current_threshold = await asyncio.to_thread(load_threshold)
        await update.message.reply_text(
            f"📊 Текущий порог фильтрации: {current_threshold:.3f}\n\n"
            f"Использование: /set_threshold <значение>\n"
            f"Пример: /set_threshold 0.6\n\n"
            f"💡 Рекомендации:\n"
            f"  • 0.0-0.5 - низкий порог (больше результатов)\n"
            f"  • 0.5-0.7 - средний порог (сбалансированный)\n"
            f"  • 0.7-1.0 - высокий порог (только очень релевантные)"
        )
        return
    
    try:
        threshold = float(context.args[0])
        
        if threshold < 0.0 or threshold > 1.0:
            await update.message.reply_text(
                "❌ Порог должен быть в диапазоне от 0.0 до 1.0"
            )
            return
        
        await asyncio.to_thread(save_threshold, threshold)
        
        await update.message.reply_text(
            f"✅ Порог фильтрации установлен: {threshold:.3f}\n\n"
            f"Теперь команда /search будет использовать этот порог для отсечения нерелевантных результатов."
        )
        
        logger.info(f"Порог фильтрации установлен пользователем {update.effective_user.id}: {threshold}")
        
    except ValueError:
        await update.message.reply_text(
            "❌ Неверный формат числа. Используйте: /set_threshold 0.6"
        )
    except Exception as e:
        logger.error(f"Ошибка в команде /set_threshold: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


async def get_threshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /get_threshold - показ текущего порога"""
    try:
        threshold = await asyncio.to_thread(load_threshold)
        await update.message.reply_text(
            f"📊 Текущий порог фильтрации: {threshold:.3f}\n\n"
            f"Используйте /set_threshold <значение> для изменения порога."
        )
    except Exception as e:
        logger.error(f"Ошибка в команде /get_threshold: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


def run_bot(token: str) -> None:
    """Запускает Telegram бота"""
    logger.info("Запуск Telegram бота")
    application = Application.builder().token(token).build()
    
    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("index", index_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("search_compare", search_compare_command))
    application.add_handler(CommandHandler("set_threshold", set_threshold_command))
    application.add_handler(CommandHandler("get_threshold", get_threshold_command))
    
    logger.info("🤖 Бот запущен и готов к работе!")
    print("🤖 Бот запущен и готов к работе!")
    print("Используйте команды /start или /help для начала")
    print(f"📝 Логи сохраняются в файл: indexing.log")
    
    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

