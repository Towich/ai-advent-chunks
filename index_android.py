"""
Скрипт для индексации Android-проекта с генерацией эмбеддингов
"""
import os
import sys
import argparse
from document_processor import process_android_project
from embedding_service import generate_embeddings_for_documents
from index_manager import save_index
from logger import setup_logger

logger = setup_logger("android_indexer")


def main():
    """Главная функция для индексации Android-проекта"""
    parser = argparse.ArgumentParser(
        description="Индексация Android-проекта для RAG-поиска"
    )
    parser.add_argument(
        "project_path",
        type=str,
        help="Путь к корневой папке Android-проекта"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Размер чанка в символах (по умолчанию: 1000)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Размер перекрытия между чанками (по умолчанию: 200)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="document_index.json",
        help="Путь к файлу для сохранения индекса (по умолчанию: document_index.json)"
    )
    
    args = parser.parse_args()
    
    project_path = os.path.abspath(args.project_path)
    
    # Проверяем существование папки
    if not os.path.exists(project_path):
        print(f"❌ Ошибка: Папка не найдена: {project_path}")
        sys.exit(1)
    
    if not os.path.isdir(project_path):
        print(f"❌ Ошибка: Указанный путь не является папкой: {project_path}")
        sys.exit(1)
    
    print(f"📱 Начало индексации Android-проекта: {project_path}")
    print(f"⚙️  Параметры: chunk_size={args.chunk_size}, overlap={args.overlap}")
    print()
    
    try:
        # Шаг 1: Обработка файлов проекта
        print("📄 Шаг 1: Обработка файлов проекта...")
        documents = process_android_project(
            project_path=project_path,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        
        if not documents:
            print("❌ Не найдено файлов для обработки")
            sys.exit(1)
        
        unique_files = len(set(doc['document'] for doc in documents))
        print(f"✅ Обработано {len(documents)} чанков из {unique_files} файлов")
        print()
        
        # Шаг 2: Генерация эмбеддингов
        print("🧠 Шаг 2: Генерация эмбеддингов через Ollama...")
        
        def progress_callback(current, total, document_name):
            percentage = (current * 100) // total if total > 0 else 0
            print(f"\r🔄 Прогресс: {current}/{total} ({percentage}%) - {document_name[:50]}", end="", flush=True)
        
        documents_with_embeddings = generate_embeddings_for_documents(
            documents,
            progress_callback=progress_callback
        )
        print()  # Новая строка после прогресса
        print(f"✅ Сгенерировано эмбеддингов: {len(documents_with_embeddings)}/{len(documents)}")
        print()
        
        # Шаг 3: Сохранение индекса
        print(f"💾 Шаг 3: Сохранение индекса в {args.output}...")
        save_index(documents_with_embeddings, index_path=args.output)
        print(f"✅ Индекс успешно сохранен в {args.output}")
        print()
        
        print("🎉 Индексация завершена успешно!")
        print(f"📊 Статистика:")
        print(f"   - Файлов обработано: {unique_files}")
        print(f"   - Чанков создано: {len(documents_with_embeddings)}")
        print(f"   - Индекс сохранен: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Индексация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка при индексации: {e}", exc_info=True)
        print(f"❌ Ошибка при индексации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

