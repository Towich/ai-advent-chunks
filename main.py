"""
Главный файл для запуска Telegram бота
"""
import os
import sys
from telegram_bot import run_bot


def main():
    """Главная функция"""
    # Получаем токен бота из переменной окружения
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token:
        print("❌ Ошибка: Не найден TELEGRAM_BOT_TOKEN в переменных окружения")
        print("\nДля запуска бота:")
        print("1. Создайте бота через @BotFather в Telegram")
        print("2. Получите токен")
        print("3. Установите переменную окружения:")
        print("   export TELEGRAM_BOT_TOKEN='ваш_токен'")
        print("4. Запустите бота: python main.py")
        sys.exit(1)
    
    # Проверяем наличие Ollama
    try:
        import requests
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama сервер доступен")
        else:
            print("⚠️  Предупреждение: Ollama сервер может быть недоступен")
    except Exception as e:
        print(f"⚠️  Предупреждение: Не удалось подключиться к Ollama: {e}")
        print("Убедитесь, что Ollama запущен на http://127.0.0.1:11434")
    
    # Запускаем бота
    run_bot(bot_token)


if __name__ == '__main__':
    main()
