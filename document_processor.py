"""
Модуль для обработки PDF документов и разбивки на чанки
"""
import os
from typing import List, Dict
from PyPDF2 import PdfReader
from logger import setup_logger

logger = setup_logger("document_processor")


def extract_text_from_pdf_batch(reader: PdfReader, pages_per_batch: int = 10):
    """
    Извлекает текст из PDF файла порциями (батчами страниц)
    
    Args:
        reader: Объект PdfReader (уже открытый)
        pages_per_batch: Количество страниц для обработки за раз
        
    Yields:
        Кортеж (порция текста, номер последней обработанной страницы, всего страниц)
    """
    total_pages = len(reader.pages)
    logger.info(f"📚 Начало извлечения текста из {total_pages} страниц (батчами по {pages_per_batch})")
    
    batch_count = 0
    for batch_start in range(0, total_pages, pages_per_batch):
        batch_count += 1
        batch_end = min(batch_start + pages_per_batch, total_pages)
        batch_text = ""
        
        logger.info(f"📄 Обработка батча #{batch_count}: страницы {batch_start + 1}-{batch_end}")
        for handler in logger.handlers:
            handler.flush()
        
        for page_num in range(batch_start, batch_end):
            try:
                if page_num == batch_start:  # Логируем первую страницу батча
                    logger.debug(f"  Извлечение текста со страницы {page_num + 1}...")
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    batch_text += page_text + "\n"
                    if page_num == batch_start:  # Логируем успех для первой страницы
                        logger.debug(f"  ✅ Страница {page_num + 1}: извлечено {len(page_text)} символов")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка при извлечении текста со страницы {page_num + 1}: {e}")
                for handler in logger.handlers:
                    handler.flush()
        
        if batch_text.strip():
            percentage = batch_end * 100 // total_pages if total_pages > 0 else 0
            logger.info(f"✅ Батч #{batch_count} готов: {len(batch_text)} символов, прогресс: {batch_end}/{total_pages} страниц ({percentage}%)")
            yield batch_text, batch_end, total_pages
        else:
            logger.warning(f"⚠️ Батч #{batch_count} пустой (страницы {batch_start + 1}-{batch_end})")
    
    logger.info(f"✅ Извлечение текста завершено: обработано {batch_count} батчей")


def split_text_into_chunks_streaming(
    text_buffer: str, 
    chunk_size: int = 1000, 
    overlap: int = 200,
    last_chunk_end: int = 0
) -> tuple[List[str], int]:
    """
    Разбивает текст на чанки с перекрытием (потоковая обработка)
    
    Args:
        text_buffer: Текст для разбивки
        chunk_size: Размер чанка в символах
        overlap: Размер перекрытия между чанками
        last_chunk_end: Позиция конца последнего чанка (для перекрытия)
        
    Returns:
        Кортеж (список чанков, позиция конца последнего чанка)
    """
    logger.debug(f"split_text_into_chunks_streaming: buffer_len={len(text_buffer)}, chunk_size={chunk_size}, overlap={overlap}, last_chunk_end={last_chunk_end}")
    
    if len(text_buffer) <= chunk_size and last_chunk_end == 0:
        result = ([text_buffer] if text_buffer.strip() else [], len(text_buffer))
        logger.debug(f"Короткий текст, возвращаем один чанк: {len(result[0])} чанков")
        return result
    
    chunks = []
    start = max(0, last_chunk_end - overlap) if last_chunk_end > 0 else 0
    logger.debug(f"Начальная позиция: start={start}")
    
    iteration = 0
    while start < len(text_buffer):
        iteration += 1
        if iteration > 1000:  # Защита от бесконечного цикла
            logger.error(f"⚠️ Превышено 1000 итераций в цикле разбивки! start={start}, buffer_len={len(text_buffer)}")
            break
        
        end = start + chunk_size
        
        # Если не последний чанк, пытаемся разбить по предложению
        if end < len(text_buffer):
            # Ищем ближайшую точку, восклицательный или вопросительный знак
            for punct in ['. ', '! ', '? ', '\n\n']:
                last_punct = text_buffer.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + len(punct)
                    break
        
        chunk = text_buffer[start:end].strip()
        if chunk:
            chunks.append(chunk)
            logger.debug(f"Итерация {iteration}: создан чанк {len(chunks)} (start={start}, end={end}, len={len(chunk)})")
        
        new_start = end - overlap
        if new_start <= start:  # Защита от зацикливания
            logger.warning(f"⚠️ new_start ({new_start}) <= start ({start}), увеличиваем на chunk_size")
            new_start = start + chunk_size
        
        start = new_start
        if start >= len(text_buffer):
            break
    
    # Возвращаем позицию конца последнего чанка для следующего батча
    last_end = start + overlap if chunks else len(text_buffer)
    logger.debug(f"Завершено: создано {len(chunks)} чанков, last_end={last_end}")
    return chunks, last_end


def process_documents_from_folder(
    folder_path: str = ".", 
    chunk_size: int = 1000, 
    overlap: int = 200,
    pages_per_batch: int = 5
) -> List[Dict]:
    """
    Обрабатывает все PDF файлы в папке и возвращает чанки с метаданными
    Обрабатывает PDF постепенно, по несколько страниц за раз
    
    Args:
        folder_path: Путь к папке с PDF файлами
        chunk_size: Размер чанка
        overlap: Размер перекрытия
        pages_per_batch: Количество страниц для обработки за раз
        
    Returns:
        Список словарей с чанками и метаданными
    """
    logger.info(f"Начало обработки документов из папки: {folder_path}")
    logger.info(f"Параметры: chunk_size={chunk_size}, overlap={overlap}, pages_per_batch={pages_per_batch}")
    
    documents = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"PDF файлы не найдены в папке {folder_path}")
        return documents
    
    logger.info(f"Найдено PDF файлов: {len(pdf_files)}")
    
    # Ищем все PDF файлы в папке
    for file_idx, filename in enumerate(pdf_files, 1):
        pdf_path = os.path.join(folder_path, filename)
        try:
            logger.info(f"[{file_idx}/{len(pdf_files)}] Обработка файла: {filename}")
            
            # Открываем PDF файл один раз
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            logger.info(f"Файл содержит {total_pages} страниц, обработка по {pages_per_batch} страниц за раз")
            # Принудительный flush для немедленного вывода
            for handler in logger.handlers:
                handler.flush()
            
            # Обрабатываем PDF постепенно, по батчам страниц
            text_buffer = ""
            last_chunk_end = 0
            chunk_counter = 0
            batch_number = 0
            
            logger.info("🔄 Начинаю обработку батчей страниц...")
            for handler in logger.handlers:
                handler.flush()
            
            # Обрабатываем по батчам
            for batch_text, pages_processed, total_pages in extract_text_from_pdf_batch(reader, pages_per_batch):
                batch_number += 1
                logger.info(f"📦 Батч #{batch_number}: получено {len(batch_text)} символов текста (страницы обработаны: {pages_processed}/{total_pages})")
                # Flush после каждого батча
                for handler in logger.handlers:
                    handler.flush()
                # Добавляем новый батч к буферу
                logger.info(f"➕ Добавляю батч #{batch_number} к буферу (текущий размер буфера: {len(text_buffer)} символов)")
                text_buffer += batch_text
                logger.info(f"📊 Размер буфера после добавления батча: {len(text_buffer)} символов")
                for handler in logger.handlers:
                    handler.flush()
                
                # Разбиваем на чанки
                logger.info(f"✂️ Начинаю разбивку буфера на чанки (размер буфера: {len(text_buffer)}, last_chunk_end: {last_chunk_end})")
                for handler in logger.handlers:
                    handler.flush()
                
                try:
                    chunks, last_chunk_end = split_text_into_chunks_streaming(
                        text_buffer, 
                        chunk_size, 
                        overlap, 
                        last_chunk_end
                    )
                    logger.info(f"✅ Разбивка завершена: создано {len(chunks)} чанков, last_chunk_end: {last_chunk_end}")
                except Exception as e:
                    logger.error(f"❌ Ошибка при разбивке на чанки: {e}", exc_info=True)
                    raise
                
                logger.info(f"✂️ Создано {len(chunks)} чанков из батча #{batch_number}")
                # Flush после создания чанков
                for handler in logger.handlers:
                    handler.flush()
                
                # Сохраняем чанки
                for chunk in chunks:
                    documents.append({
                        'document': filename,
                        'chunk_id': chunk_counter,
                        'text': chunk,
                        'total_chunks': 0  # Будет обновлено позже
                    })
                    chunk_counter += 1
                
                logger.debug(f"Всего чанков создано: {chunk_counter}")
                
                # Оставляем только перекрытие для следующего батча
                # Это позволяет не держать весь текст в памяти
                if last_chunk_end > overlap:
                    text_buffer = text_buffer[last_chunk_end - overlap:]
                    last_chunk_end = overlap
                else:
                    # Если перекрытие больше буфера, оставляем весь буфер
                    text_buffer = text_buffer[max(0, len(text_buffer) - overlap):]
                    last_chunk_end = min(overlap, len(text_buffer))
                
                # Логируем прогресс каждые 10 страниц или каждые 25 чанков
                if pages_processed % 10 == 0 or chunk_counter % 25 == 0:
                    percentage = pages_processed * 100 // total_pages if total_pages > 0 else 0
                    logger.info(
                        f"📄 Прогресс: {pages_processed}/{total_pages} страниц ({percentage}%), "
                        f"создано {chunk_counter} чанков"
                    )
            
            # Обрабатываем остаток буфера
            if text_buffer.strip():
                chunks, _ = split_text_into_chunks_streaming(
                    text_buffer, 
                    chunk_size, 
                    overlap, 
                    last_chunk_end
                )
                for chunk in chunks:
                    documents.append({
                        'document': filename,
                        'chunk_id': chunk_counter,
                        'text': chunk,
                        'total_chunks': 0
                    })
                    chunk_counter += 1
            
            # Обновляем total_chunks для всех чанков этого документа
            total_chunks = chunk_counter
            for doc in documents:
                if doc['document'] == filename:
                    doc['total_chunks'] = total_chunks
            
            logger.info(f"Файл {filename} успешно обработан: {total_chunks} чанков")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {filename}: {e}", exc_info=True)
    
    logger.info(f"Обработка завершена. Всего создано {len(documents)} чанков из {len(pdf_files)} файлов")
    return documents

