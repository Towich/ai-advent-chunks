"""
Модуль для обработки PDF и Markdown документов и разбивки на чанки
"""
import os
import re
from typing import List, Dict, Tuple
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
    seen_chunks = set()  # Множество для отслеживания уже добавленных чанков
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
            # Проверяем на дубликаты перед добавлением
            if chunk not in seen_chunks:
                chunks.append(chunk)
                seen_chunks.add(chunk)
                logger.debug(f"Итерация {iteration}: создан чанк {len(chunks)} (start={start}, end={end}, len={len(chunk)})")
            else:
                logger.debug(f"Итерация {iteration}: пропущен дубликат чанка (start={start}, end={end})")
        
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


def parse_markdown_headers(content: str) -> List[Tuple[int, int, int, str]]:
    """
    Парсит markdown файл и находит все заголовки с их позициями
    
    Args:
        content: Содержимое markdown файла
        
    Returns:
        Список кортежей (уровень_заголовка, позиция_начала, позиция_конца, текст_заголовка)
        Уровень: 1 для #, 2 для ##, и т.д.
    """
    headers = []
    lines = content.split('\n')
    current_pos = 0
    
    for i, line in enumerate(lines):
        # Проверяем заголовки в формате # Заголовок
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            header_text = match.group(2).strip()
            start_pos = current_pos
            # Конец заголовка - начало следующей строки
            end_pos = current_pos + len(line)
            headers.append((level, start_pos, end_pos, header_text))
        
        current_pos += len(line) + 1  # +1 для символа новой строки
    
    return headers


def split_markdown_by_headers(
    content: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100
) -> List[Dict[str, str]]:
    """
    Умное разбиение markdown файла на чанки на основе заголовков
    
    Args:
        content: Содержимое markdown файла
        chunk_size: Максимальный размер чанка
        overlap: Размер перекрытия между чанками
        min_chunk_size: Минимальный размер чанка (если секция меньше, объединяем с предыдущей)
        
    Returns:
        Список словарей с чанками и метаданными (text, header_context)
    """
    headers = parse_markdown_headers(content)
    chunks = []
    
    if not headers:
        # Если заголовков нет, используем обычное разбиение
        logger.info("Заголовки не найдены, используем обычное разбиение")
        text_chunks, _ = split_text_into_chunks_streaming(content, chunk_size, overlap)
        for chunk_text in text_chunks:
            chunks.append({
                'text': chunk_text,
                'header_context': ''
            })
        return chunks
    
    logger.info(f"Найдено {len(headers)} заголовков в markdown файле")
    
    # Создаем секции на основе заголовков
    sections = []
    for i, (level, start_pos, end_pos, header_text) in enumerate(headers):
        # Определяем конец секции (начало следующего заголовка того же или более высокого уровня)
        section_end = len(content)
        
        for j in range(i + 1, len(headers)):
            next_level, next_start, _, _ = headers[j]
            # Если следующий заголовок того же или более высокого уровня, это конец секции
            if next_level <= level:
                section_end = next_start
                break
        
        # Извлекаем текст секции (пропускаем пустые строки после заголовка)
        section_text = content[end_pos:section_end]
        # Убираем начальные пустые строки
        section_text = section_text.lstrip('\n\r').strip()
        
        if section_text:
            sections.append({
                'level': level,
                'header': header_text,
                'text': section_text,
                'start': end_pos,
                'end': section_end
            })
    
    # Если есть текст до первого заголовка, добавляем его как отдельную секцию
    if headers and headers[0][1] > 0:
        pre_header_text = content[:headers[0][1]].strip()
        if pre_header_text:
            sections.insert(0, {
                'level': 0,
                'header': 'Введение',
                'text': pre_header_text,
                'start': 0,
                'end': headers[0][1]
            })
    
    # Если секций нет, используем весь текст
    if not sections:
        sections.append({
            'level': 0,
            'header': '',
            'text': content,
            'start': 0,
            'end': len(content)
        })
    
    logger.info(f"Создано {len(sections)} секций на основе заголовков")
    
    # Разбиваем секции на чанки
    current_header_path = []  # Путь заголовков для контекста
    seen_texts = set()  # Множество для отслеживания уже добавленных текстов
    
    for section in sections:
        section_text = section['text']
        header = section['header']
        level = section['level']
        
        # Обновляем путь заголовков
        # Удаляем заголовки более глубокого уровня
        current_header_path = [h for h in current_header_path if h[0] < level]
        # Добавляем текущий заголовок
        current_header_path.append((level, header))
        
        # Формируем контекст заголовков (уже отсортированы по уровню)
        header_context = ' > '.join([h[1] for h in current_header_path])
        
        # Если секция маленькая, создаем один чанк
        if len(section_text) <= chunk_size:
            # Проверяем на дубликаты перед добавлением
            text_normalized = section_text.strip()
            if text_normalized and text_normalized not in seen_texts:
                chunks.append({
                    'text': section_text,
                    'header_context': header_context
                })
                seen_texts.add(text_normalized)
            else:
                logger.debug(f"Пропущен дубликат чанка (секция: {header})")
        else:
            # Если секция большая, разбиваем её на подчанки
            sub_chunks, _ = split_text_into_chunks_streaming(
                section_text,
                chunk_size,
                overlap
            )
            
            for sub_chunk in sub_chunks:
                # Проверяем на дубликаты перед добавлением
                text_normalized = sub_chunk.strip()
                if text_normalized and text_normalized not in seen_texts:
                    chunks.append({
                        'text': sub_chunk,
                        'header_context': header_context
                    })
                    seen_texts.add(text_normalized)
                else:
                    logger.debug(f"Пропущен дубликат подчанка (секция: {header})")
    
    duplicates_removed = len(sections) * 2 - len(chunks)  # Примерная оценка
    logger.info(f"Создано {len(chunks)} уникальных чанков из markdown файла (дубликаты удалены)")
    return chunks


def process_markdown_file(
    file_path: str,
    filename: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict]:
    """
    Обрабатывает один markdown файл и возвращает чанки с метаданными
    
    Args:
        file_path: Полный путь к markdown файлу
        filename: Имя файла
        chunk_size: Размер чанка
        overlap: Размер перекрытия
        
    Returns:
        Список словарей с чанками и метаданными
    """
    logger.info(f"Обработка markdown файла: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Файл прочитан: {len(content)} символов")
        
        # Умное разбиение на чанки по заголовкам
        markdown_chunks = split_markdown_by_headers(content, chunk_size, overlap)
        
        documents = []
        for chunk_idx, chunk_data in enumerate(markdown_chunks):
            documents.append({
                'document': filename,
                'chunk_id': chunk_idx,
                'text': chunk_data['text'],
                'header_context': chunk_data.get('header_context', ''),
                'total_chunks': len(markdown_chunks)
            })
        
        logger.info(f"Markdown файл {filename} успешно обработан: {len(documents)} чанков")
        return documents
        
    except Exception as e:
        logger.error(f"Ошибка при обработке markdown файла {filename}: {e}", exc_info=True)
        return []


def process_documents_from_folder(
    folder_path: str = "./docs",
    chunk_size: int = 500,
    overlap: int = 50,
    pages_per_batch: int = 5
) -> List[Dict]:
    """
    Обрабатывает все PDF и Markdown файлы в папке и возвращает чанки с метаданными
    Обрабатывает PDF постепенно, по несколько страниц за раз
    Markdown файлы обрабатываются с умным разбиением по заголовкам
    
    Args:
        folder_path: Путь к папке с документами
        chunk_size: Размер чанка
        overlap: Размер перекрытия
        pages_per_batch: Количество страниц для обработки за раз (только для PDF)
        
    Returns:
        Список словарей с чанками и метаданными
    """
    logger.info(f"Начало обработки документов из папки: {folder_path}")
    logger.info(f"Параметры: chunk_size={chunk_size}, overlap={overlap}, pages_per_batch={pages_per_batch}")
    
    documents = []
    
    # Ищем PDF и MD файлы
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    md_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.md')]
    
    if not pdf_files and not md_files:
        logger.warning(f"PDF и MD файлы не найдены в папке {folder_path}")
        return documents
    
    logger.info(f"Найдено PDF файлов: {len(pdf_files)}, MD файлов: {len(md_files)}")
    
    # Обрабатываем PDF файлы
    
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
            seen_chunks_pdf = set()  # Множество для отслеживания уже добавленных чанков для этого PDF
            
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
                
                # Сохраняем чанки (с проверкой на дубликаты)
                for chunk in chunks:
                    chunk_normalized = chunk.strip()
                    if chunk_normalized and chunk_normalized not in seen_chunks_pdf:
                        documents.append({
                            'document': filename,
                            'chunk_id': chunk_counter,
                            'text': chunk,
                            'total_chunks': 0  # Будет обновлено позже
                        })
                        seen_chunks_pdf.add(chunk_normalized)
                        chunk_counter += 1
                    else:
                        logger.debug(f"Пропущен дубликат чанка для PDF {filename}")
                
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
                    chunk_normalized = chunk.strip()
                    if chunk_normalized and chunk_normalized not in seen_chunks_pdf:
                        documents.append({
                            'document': filename,
                            'chunk_id': chunk_counter,
                            'text': chunk,
                            'total_chunks': 0
                        })
                        seen_chunks_pdf.add(chunk_normalized)
                        chunk_counter += 1
                    else:
                        logger.debug(f"Пропущен дубликат чанка для PDF {filename} (остаток буфера)")
            
            # Обновляем total_chunks для всех чанков этого документа
            total_chunks = chunk_counter
            for doc in documents:
                if doc['document'] == filename:
                    doc['total_chunks'] = total_chunks
            
            logger.info(f"Файл {filename} успешно обработан: {total_chunks} чанков")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {filename}: {e}", exc_info=True)
    
    # Обрабатываем Markdown файлы
    for file_idx, filename in enumerate(md_files, 1):
        md_path = os.path.join(folder_path, filename)
        try:
            logger.info(f"[{file_idx}/{len(md_files)}] Обработка markdown файла: {filename}")
            md_documents = process_markdown_file(md_path, filename, chunk_size, overlap)
            documents.extend(md_documents)
        except Exception as e:
            logger.error(f"Ошибка при обработке markdown файла {filename}: {e}", exc_info=True)
    
    total_files = len(pdf_files) + len(md_files)
    logger.info(f"Обработка завершена. Всего создано {len(documents)} чанков из {total_files} файлов ({len(pdf_files)} PDF, {len(md_files)} MD)")
    return documents

