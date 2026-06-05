"""HTTP-сервер для приёма Callback API запросов от VK.

VK отправляет POST-запросы на configured URL. Сервер обрабатывает:
- confirmation: возвращает код подтверждения из .env
- message_new: передаёт сообщение оркестратору и отвечает через VK API
"""

import json
from aiohttp import web
from vkbottle import Bot
from src.agents.orchestrator import Orchestrator
from src.utils.config import settings
from src.utils.logger import logger


class VKCallbackServer:
    """Сервер для обработки VK Callback API."""

    def __init__(self, bot: Bot, orchestrator: Orchestrator):
        self._bot = bot
        self._orchestrator = orchestrator
        self._app = web.Application()
        self._app.router.add_post("/", self._handle_callback)
        self._app.router.add_get("/", self._handle_health)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Healthcheck для ngrok и мониторинга."""
        return web.Response(text="Bronoskins Bot is running")

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Обработка входящего callback от VK."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            logger.error("Callback: невалидный JSON")
            return web.Response(text="bad request", status=400)

        event_type = body.get("type", "")
        group_id = body.get("group_id", "")
        secret = body.get("secret", "")

        logger.debug("Callback: type={}, group_id={}", event_type, group_id)

        # Проверка секретного ключа (если задан)
        if settings.vk_callback_secret and secret != settings.vk_callback_secret:
            logger.warning("Callback: неверный secret")
            return web.Response(text="unauthorized", status=403)

        # Подтверждение сервера
        if event_type == "confirmation":
            confirmation = settings.vk_callback_confirmation
            if not confirmation:
                logger.error("Callback: VK_CALLBACK_CONFIRMATION не задан в .env!")
                return web.Response(text="confirmation not configured", status=500)
            logger.info("Callback: подтверждение сервера (group_id={})", group_id)
            return web.Response(text=confirmation)

        # Новое сообщение
        if event_type == "message_new":
            return await self._handle_message(body)

        # Остальные события — просто OK
        logger.debug("Callback: пропущено событие типа '{}'", event_type)
        return web.Response(text="ok")

    async def _handle_message(self, body: dict) -> web.Response:
        """Обработка нового сообщения из VK."""
        obj = body.get("object", {})
        message_data = obj.get("message", obj)

        user_id = message_data.get("from_id", 0)
        peer_id = message_data.get("peer_id", 0)
        text = message_data.get("text", "")

        if not text:
            return web.Response(text="ok")

        logger.info(
            "Callback: новое сообщение от user_id={}, текст='{}'",
            user_id,
            text[:100],
        )

        # Получаем имя пользователя
        user_name = ""
        try:
            users = await self._bot.api.users.get(user_ids=[user_id])
            if users:
                user_name = users[0].first_name
        except Exception as e:
            logger.debug("Callback: не удалось получить имя: {}", e)

        # Обрабатываем через оркестратор
        try:
            response = await self._orchestrator.process(
                user_query=text,
                user_id=user_id,
                user_name=user_name,
            )
        except Exception as e:
            logger.error("Callback: ошибка оркестратора: {}", e)
            response = (
                "Извините, произошла техническая ошибка. "
                "Пожалуйста, попробуйте позже. 😊"
            )

        # Отправляем ответ через VK API
        try:
            await self._bot.api.messages.send(
                peer_id=peer_id,
                message=response,
                random_id=0,
            )
            logger.info(
                "Callback: ответ отправлен user_id={} ({} символов)",
                user_id,
                len(response),
            )
        except Exception as e:
            logger.error("Callback: ошибка отправки ответа: {}", e)

        return web.Response(text="ok")

    async def start(self):
        """Асинхронный запуск сервера (внутри существующего event loop)."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            host=settings.vk_callback_host,
            port=settings.vk_callback_port,
        )
        await site.start()
        logger.info(
            "Callback сервер запущен на http://{}:{}",
            settings.vk_callback_host,
            settings.vk_callback_port,
        )
        # Держим сервер живым бесконечно
        import asyncio
        await asyncio.Event().wait()

    def run(self):
        """Синхронный запуск (создаёт собственный event loop)."""
        logger.info(
            "Callback сервер запущен на http://{}:{}",
            settings.vk_callback_host,
            settings.vk_callback_port,
        )
        web.run_app(
            self._app,
            host=settings.vk_callback_host,
            port=settings.vk_callback_port,
            print=lambda *a, **kw: None,
        )
