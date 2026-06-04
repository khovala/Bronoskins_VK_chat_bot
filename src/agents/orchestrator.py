"""Оркестратор мультиагентной RAG-системы.

Координирует работу всех агентов:
1. Принимает сообщение пользователя
2. Запускает RAG-поиск для получения контекста
3. Оценивает релевантность через Critic Agent
4. Генерирует ответ через Sales Agent (LLM) с учётом контекста
5. При необходимости вызывает YCLIENTS для записи

Реализован как кастомный оркестратор (без CrewAI) для максимального
контроля и асинхронности. CrewAI можно подключить опционально.
"""

from typing import List
from src.agents.rag_agent import RagAgent
from src.agents.critic_agent import CriticAgent
from src.agents.yclients_agent import YClientsAgent
from src.agents.sales_agent import SYSTEM_PROMPT_SALES
from src.llm.base import BaseLLM
from src.rag.vector_store import VectorStore
from src.utils.config import settings
from src.utils.logger import logger


class Orchestrator:
    """Главный оркестратор мультиагентной системы Bronoskins."""

    def __init__(
        self,
        llm_client: BaseLLM,
        vector_store: VectorStore,
        yclients_agent: YClientsAgent | None = None,
    ):
        self._llm = llm_client
        self._rag_agent = RagAgent(
            vector_store, top_k=settings.search_top_k
        )
        self._critic = CriticAgent(llm_client=llm_client)
        self._yclients = yclients_agent

        # История диалогов по пользователям
        self._histories: dict[int, list[str]] = {}

    def _get_history(self, user_id: int) -> list[str]:
        """Получение истории диалога пользователя."""
        return self._histories.get(user_id, [])

    def _save_history(self, user_id: int, messages: list[str]):
        """Сохранение истории диалога (не более max_history_messages)."""
        max_len = settings.max_history_messages
        self._histories[user_id] = messages[-max_len:]

    async def process(
        self,
        user_query: str,
        user_id: int,
        user_name: str = "",
    ) -> str:
        """Обработка сообщения пользователя.

        Основной пайплайн:
        1. Поиск контекста в RAG
        2. Проверка релевантности (с повторными попытками)
        3. Генерация ответа через LLM с системным промптом Sales Agent
        """
        logger.info(
            "Оркестратор: запрос от user_id={}, "
            "текст='{}'",
            user_id,
            user_query[:100],
        )

        # 1. Поиск в RAG с проверкой релевантности
        context = await self._retrieve_with_critic(user_query)

        # 2. Формирование истории для контекста LLM
        history = self._get_history(user_id)
        history_text = self._format_history(history)

        # 3. Формирование полного промпта
        full_prompt = self._build_prompt(
            user_query=user_query,
            user_name=user_name,
            context=context,
            history_text=history_text,
        )

        # 4. Генерация ответа
        try:
            response = await self._llm.generate(
                prompt=full_prompt,
                system_prompt=SYSTEM_PROMPT_SALES,
                temperature=0.7,
                max_tokens=1000,
            )
        except Exception as e:
            logger.error("Оркестратор: ошибка генерации ответа: {}", e)
            response = (
                "Извините, произошла техническая ошибка. "
                "Пожалуйста, попробуйте позже или напишите нам в личные сообщения. 😊"
            )

        # 5. Сохранение истории
        history.append(f"Клиент: {user_query}")
        history.append(f"Бот: {response}")
        self._save_history(user_id, history)

        logger.info(
            "Оркестратор: ответ сгенерирован ({} символов) для user_id={}",
            len(response),
            user_id,
        )
        return response

    async def _retrieve_with_critic(self, query: str) -> str:
        """Поиск в RAG с оценкой релевантности и повторными попытками."""
        max_attempts = 3
        current_query = query

        for attempt in range(max_attempts):
            # Поиск
            documents = self._rag_agent.search(current_query)

            if not documents:
                logger.debug(
                    "Оркестратор: RAG-поиск не дал результатов "
                    "(попытка {}/{})",
                    attempt + 1,
                    max_attempts,
                )
                if attempt < max_attempts - 1:
                    refined = self._critic.generate_refined_query(
                        query, []
                    )
                    if refined:
                        current_query = refined
                        continue
                return ""

            # Оценка релевантности
            is_relevant, score = self._critic.evaluate(current_query, documents)

            if is_relevant:
                logger.debug(
                    "Оркестратор: RAG-результаты релевантны "
                    "(score={:.3f}, попытка {})",
                    score,
                    attempt + 1,
                )
                return self._rag_agent.format_context(documents)

            # Если нерелевантны — пробуем уточнить запрос
            if attempt < max_attempts - 1:
                refined = self._critic.generate_refined_query(
                    query, documents
                )
                if refined:
                    current_query = refined
                    continue

            # Последняя попытка — возвращаем что есть
            logger.debug(
                "Оркестратор: низкая релевантность, "
                "возвращаем результаты как есть (score={:.3f})",
                score,
            )
            return self._rag_agent.format_context(documents)

        return ""

    def _build_prompt(
        self,
        user_query: str,
        user_name: str,
        context: str,
        history_text: str,
    ) -> str:
        """Формирование финального промпта для LLM."""
        parts = []

        # Контекст из базы знаний
        if context:
            parts.append(f"📚 Информация из базы знаний Bronoskins:\n{context}")

        # История диалога
        if history_text:
            parts.append(f"💬 История диалога:\n{history_text}")

        # Текущий запрос
        if user_name:
            parts.append(f"👤 Клиент {user_name} пишет: {user_query}")
        else:
            parts.append(f"👤 Клиент пишет: {user_query}")

        parts.append(
            "\nСгенерируй ответ как ИИ-консультант Bronoskins. "
            "Следуй маркетинговым инструкциям, предложи запись."
        )

        return "\n\n".join(parts)

    @staticmethod
    def _format_history(history: List[str]) -> str:
        """Форматирование истории диалога для промпта."""
        if not history:
            return ""
        return "\n".join(history[-10:])

    async def get_available_slots_for_service(
        self, service_name: str
    ) -> str:
        """Получение слотов через YCLIENTS по ВСЕМ филиалам."""
        if self._yclients is None or not self._yclients.is_configured():
            return (
                "Сервис записи временно недоступен. "
                "Пожалуйста, напишите нам для уточнения времени."
            )

        try:
            from datetime import date, timedelta
            today = date.today()
            week_later = today + timedelta(days=7)

            all_slots = await self._yclients.search_all_filials_for_slots(
                service_name=service_name,
                date_from=today,
                date_to=week_later,
            )

            # Формируем ответ: группируем слоты по филиалам
            has_any = False
            result = ["Доступное время для записи:\n"]

            for filial_name, slots in all_slots.items():
                address = self._yclients.get_location(filial_name) or filial_name
                if slots:
                    has_any = True
                    result.append(f"🏢 {filial_name} ({address}):")
                    for slot in slots[:5]:
                        date_str = slot.get("_date", "?")
                        time_str = slot.get("time", slot.get("datetime", "?"))
                        result.append(f"   📅 {date_str} в {time_str}")
                else:
                    result.append(f"🏢 {filial_name} ({address}): слотов нет")

            if not has_any:
                return (
                    "На ближайшую неделю свободных слотов нет ни в одном филиале. "
                    "Но вы можете написать нам, и мы подберём удобное время!"
                )

            result.append("\nВ каком филиале и в какое время вам удобно?")
            return "\n".join(result)

        except Exception as e:
            logger.error("Оркестратор: ошибка получения слотов: {}", e)
            return "Произошла ошибка при проверке свободного времени. Давайте я помогу вам записаться вручную."

    async def book_client(
        self,
        client_name: str,
        client_phone: str,
        service_name: str,
        datetime_str: str,
        filial_name: str = "",
    ) -> str:
        """Создание записи клиента через YCLIENTS в указанном филиале."""
        if self._yclients is None or not self._yclients.is_configured():
            return (
                "Сервис записи временно недоступен. "
                "Наш менеджер свяжется с вами в ближайшее время!"
            )

        try:
            company_id = self._yclients.get_company_id(filial_name) if filial_name else None
            if company_id is None:
                return (
                    "Уточните, пожалуйста, в каком филиале вам удобно записаться: "
                    + ", ".join(self._yclients.filial_names)
                    + "?"
                )

            service_id = await self._yclients._find_service_id(
                service_name, company_id
            )
            if service_id is None:
                return f"Услуга '{service_name}' не найдена в филиале '{filial_name}'."

            result = await self._yclients.book_appointment(
                company_id=company_id,
                client_name=client_name,
                client_phone=client_phone,
                service_id=service_id,
                appointment_datetime=datetime_str,
            )

            if result:
                locations = self._yclients.get_locations_text()
                return (
                    f"✅ Отлично! Запись подтверждена.\n\n"
                    f"👤 Имя: {client_name}\n"
                    f"📞 Телефон: {client_phone}\n"
                    f"📅 Дата и время: {datetime_str}\n"
                    f"🛠 Услуга: {service_name}\n"
                    f"🏢 Филиал: {filial_name}\n\n"
                    f"📍 Адреса установки:\n{locations}\n\n"
                    f"Будем ждать вас! 😊"
                )

            return "Не удалось создать запись. Попробуйте позже или обратитесь к администратору."

        except Exception as e:
            logger.error("Оркестратор: ошибка создания записи: {}", e)
            return "Произошла ошибка при создании записи. Наш менеджер свяжется с вами."
