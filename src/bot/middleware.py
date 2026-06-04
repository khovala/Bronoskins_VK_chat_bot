"""Middleware для VK-бота: логирование, обработка ошибок."""

from vkbottle import BaseMiddleware
from vkbottle.bot import Message
from src.utils.logger import logger


class LoggingMiddleware(BaseMiddleware[Message]):
    """Middleware для логирования входящих сообщений."""

    async def pre(self):
        """Действие перед обработкой сообщения."""
        msg = self.event
        logger.debug(
            "Middleware.pre: сообщение от user_id={}, "
            "peer_id={}, текст='{}' (запуск обработчика)",
            msg.from_id,
            msg.peer_id,
            str(msg.text)[:80],
        )

    async def post(self):
        """Действие после обработки сообщения."""
        msg = self.event
        logger.debug(
            "Middleware.post: сообщение от user_id={} обработано",
            msg.from_id,
        )
