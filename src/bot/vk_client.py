"""Инициализация VK-бота на фреймворке vkbottle.

Используется Bots LongPoll API для простоты запуска.
Токен бота загружается из переменной окружения VK_TOKEN.
"""

from vkbottle import Bot
from src.utils.config import settings
from src.utils.logger import logger


def create_bot() -> Bot:
    """Создание и настройка экземпляра VK-бота."""
    if not settings.vk_token:
        logger.error(
            "VK_TOKEN не задан в .env! "
            "Бот не сможет подключиться к VK API."
        )
        raise RuntimeError(
            "VK_TOKEN не задан. "
            "Создайте .env файл с VK_TOKEN=ваш_токен_группы"
        )

    bot = Bot(token=settings.vk_token)
    logger.info("VK-бот инициализирован (токен загружен)")
    return bot
