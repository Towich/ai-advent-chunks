"""
Telegram бот для обработки документов и создания индекса с эмбеддингами
"""
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from document_processor import process_documents_from_folder
from embedding_service import generate_embeddings_for_documents
from index_manager import save_index, get_index_stats
from logger import setup_logger

logger = setup_logger("telegram_bot")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    logger.info(f"Получена команда /start от пользователя {update.effective_user.id}")
    welcome_message = """
🤖 Бот для индексации документов с эмбеддингами

Доступные команды:
/start - Показать это сообщение
/index - Создать индекс из PDF файлов в папке проекта
/stats - Показать статистику по индексу
/help - Показать справку
"""
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help"""
    help_text = """
📚 Справка по использованию бота:

/index - Запускает процесс индексации:
  1. Находит все PDF файлы в папке проекта
  2. Извлекает текст из PDF
  3. Разбивает на чанки
  4. Генерирует эмбеддинги через Ollama (nomic-embed-text)
  5. Сохраняет индекс в document_index.json

/stats - Показывает статистику по существующему индексу

⚠️ Убедитесь, что:
  - Ollama запущен на http://127.0.0.1:11434
  - Модель nomic-embed-text установлена в Ollama
  - PDF файлы находятся в папке проекта
"""
    await update.message.reply_text(help_text)


async def index_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /index - создание индекса"""
    chat_id = update.effective_chat.id
    logger.info(f"Получена команда /index от пользователя {update.effective_user.id}")
    
    await update.message.reply_text("🔄 Начинаю обработку документов...")
    
    try:
        # Обработка документов (в отдельном потоке, чтобы не блокировать)
        await update.message.reply_text("📄 Ищу PDF файлы и извлекаю текст...")
        logger.info("Начало обработки PDF файлов")
        # Обрабатываем по 5 страниц за раз для экономии памяти
        documents = await asyncio.to_thread(process_documents_from_folder)
        
        if not documents:
            logger.warning("PDF файлы не найдены в папке проекта")
            await update.message.reply_text("❌ PDF файлы не найдены в папке проекта")
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


def run_bot(token: str) -> None:
    """Запускает Telegram бота"""
    logger.info("Запуск Telegram бота")
    application = Application.builder().token(token).build()
    
    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("index", index_command))
    application.add_handler(CommandHandler("stats", stats_command))
    
    logger.info("🤖 Бот запущен и готов к работе!")
    print("🤖 Бот запущен и готов к работе!")
    print("Используйте команды /start или /help для начала")
    print(f"📝 Логи сохраняются в файл: indexing.log")
    
    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

