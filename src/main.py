"""Точка входа приложения Bronoskins Bot.

Запуск: python -m src.main

Порядок инициализации:
1. Загрузка конфигурации (.env)
2. Настройка логирования (loguru)
3. Инициализация векторной БД (Chroma/Qdrant)
4. Инициализация LLM-клиента (YandexGPT)
5. Инициализация YCLIENTS-агента
6. Создание оркестратора
7. Создание и запуск VK-бота
"""

import asyncio
import signal
import sys

from src.utils.config import settings
from src.utils.logger import logger
from src.rag.vector_store import VectorStore
from src.llm.yandexgpt import YandexGPTClient
from src.agents.orchestrator import Orchestrator
from src.agents.yclients_agent import YClientsAgent
from src.agents.tools import set_vector_store, set_yclients_agent
from src.bot.vk_client import create_bot
from src.bot.handlers import register_handlers
from src.bot.middleware import LoggingMiddleware


async def main():
    """Главная функция запуска приложения."""
    logger.info("=" * 60)
    logger.info("Запуск Bronoskins Bot")
    logger.info("=" * 60)

    # 1. Инициализация векторной БД
    logger.info("Инициализация векторной БД (backend={})...",
                 settings.vector_store_backend)
    vector_store = VectorStore()
    try:
        vector_store.initialize()
        logger.info("Векторная БД готова")
    except Exception as e:
        logger.error("Ошибка инициализации векторной БД: {}", e)
        logger.warning(
            "Бот запускается без RAG-поиска. "
            "Запустите indexer.py для индексации документов."
        )
        # Создаём заглушку, чтобы не падать
        vector_store = None

    # 2. Инициализация LLM-клиента
    logger.info("Инициализация YandexGPT клиента...")
    try:
        llm_client = YandexGPTClient()
        logger.info("YandexGPT клиент готов (модель: {})",
                     settings.yandexgpt_model)
    except Exception as e:
        logger.error("Ошибка инициализации YandexGPT: {}", e)
        logger.error(
            "Проверьте YANDEXGPT_API_KEY и YANDEXGPT_FOLDER_ID в .env"
        )
        sys.exit(1)

    # 3. Инициализация YCLIENTS-агента
    yclients_agent = None
    import json
    has_filials = bool(settings.yclients_token and settings.yclients_filials and
                       settings.yclients_filials != "{}")
    if has_filials:
        logger.info("Инициализация YCLIENTS агента...")
        try:
            yclients_agent = YClientsAgent()
            if yclients_agent.is_configured():
                logger.info("YCLIENTS агент готов ({} филиалов)",
                             len(yclients_agent.filial_names))
            else:
                logger.warning("YCLIENTS: агент создан, но филиалы не настроены")
                yclients_agent = None
        except Exception as e:
            logger.error("Ошибка инициализации YCLIENTS: {}", e)
            logger.warning("Бот запускается без интеграции с YCLIENTS")
    else:
        logger.warning(
            "YCLIENTS_TOKEN или YCLIENTS_FILIALS не заданы. "
            "Интеграция с YCLIENTS отключена."
        )

    # 4. Установка глобальных инструментов для CrewAI (опционально)
    if vector_store:
        set_vector_store(vector_store)
    if yclients_agent:
        set_yclients_agent(yclients_agent)

    # 5. Создание оркестратора
    logger.info("Создание оркестратора...")
    orchestrator = Orchestrator(
        llm_client=llm_client,
        vector_store=vector_store,
        yclients_agent=yclients_agent,
    )
    logger.info("Оркестратор готов")

    # 6. Создание VK-бота
    logger.info("Создание VK-бота...")
    bot = create_bot()

    # Регистрация middleware
    bot.labeler.message_view.register_middleware(LoggingMiddleware)

    # Регистрация обработчиков
    register_handlers(bot, orchestrator)
    logger.info("Обработчики зарегистрированы")

    # 7. Graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info("Получен сигнал {}, завершение работы...", sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("Бот Bronoskins запущен и слушает сообщения...")
    logger.info("=" * 60)

    # 8. Запуск бота (vkbottle сам управляет event loop'ом)
    try:
        # vkbottle 4.x: run() — синхронный метод, сам создаёт и управляет event loop
        await bot.run_polling()
    except Exception as e:
        logger.error("Ошибка в работе бота: {}", e)
    finally:
        logger.info("Бот остановлен. Завершение работы...")
        # Здесь можно добавить cleanup ресурсов


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
