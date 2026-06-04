"""Обработчики сообщений VK-бота.

Основной обработчик принимает входящие сообщения,
вызывает оркестратора и отправляет ответ пользователю.
"""

from vkbottle import Bot
from vkbottle.bot import Message
from src.agents.orchestrator import Orchestrator
from src.utils.logger import logger


def register_handlers(bot: Bot, orchestrator: Orchestrator):
    """Регистрация обработчиков сообщений на боте."""

    @bot.on.message()
    async def handle_message(message: Message):
        """Главный обработчик входящих сообщений."""
        user_id = message.from_id

        # Игнорируем сообщения без текста
        if not message.text:
            logger.debug("Пропущено пустое сообщение от user_id={}", user_id)
            return

        # Получаем имя пользователя из профиля VK (опционально)
        user_name = ""
        try:
            users = await bot.api.users.get(user_ids=[user_id])
            if users:
                user_name = users[0].first_name
        except Exception as e:
            logger.debug("Не удалось получить имя пользователя: {}", e)

        logger.info(
            "Входящее сообщение от {} (id={}): {}",
            user_name or "клиент",
            user_id,
            message.text[:100],
        )

        try:
            # Вызываем оркестратора для обработки сообщения
            response = await orchestrator.process(
                user_query=message.text,
                user_id=user_id,
                user_name=user_name,
            )

            # Отправляем ответ
            await message.answer(response)
            logger.info(
                "Ответ отправлен пользователю {} ({} символов)",
                user_id,
                len(response),
            )

        except Exception as e:
            logger.error(
                "Ошибка при обработке сообщения от user_id={}: {}",
                user_id,
                e,
                exc_info=True,
            )
            try:
                await message.answer(
                    "Извините, произошла техническая ошибка. "
                    "Пожалуйста, попробуйте позже или напишите нам "
                    "в личные сообщения. 😊"
                )
            except Exception as send_error:
                logger.error(
                    "Не удалось отправить сообщение об ошибке: {}",
                    send_error,
                )
