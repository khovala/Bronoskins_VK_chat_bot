"""Инструменты для CrewAI-агентов.

Содержит search_tool (поиск в RAG) и yclients_tool (запись через YCLIENTS).
Инструменты оборачивают вызовы соответствующих агентов в формат CrewAI Tool.
"""

from typing import Optional
from langchain.tools import tool
from src.rag.vector_store import VectorStore
from src.utils.logger import logger


# Глобальные ссылки на компоненты (устанавливаются при инициализации приложения)
_vector_store: Optional[VectorStore] = None
_yclients_agent = None


def set_vector_store(store: VectorStore):
    """Установка глобальной ссылки на VectorStore для инструментов."""
    global _vector_store
    _vector_store = store


def set_yclients_agent(agent):
    """Установка глобальной ссылки на YClientsAgent для инструментов."""
    global _yclients_agent
    _yclients_agent = agent


@tool
def search_knowledge_base(query: str) -> str:
    """Поиск информации в базе знаний Bronoskins.

    Используй этот инструмент когда нужно найти:
    - Информацию о продуктах и услугах
    - Скрипты ответов на вопросы
    - Базу возражений
    - Инструкции по гарантии
    - Цены и условия

    Args:
        query: Поисковый запрос на русском языке.

    Returns:
        Релевантные фрагменты из базы знаний.
    """
    if _vector_store is None:
        return "База знаний не инициализирована."

    try:
        docs = _vector_store.search(query, k=5)
        if not docs:
            return "В базе знаний не найдено информации по вашему запросу."

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "неизвестно")
            parts.append(f"[Источник: {source}]\n{doc.page_content}")

        logger.debug("Поиск в БЗ: запрос '{}' — найдено {} документов",
                      query[:60], len(docs))
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.error("Ошибка поиска в БЗ: {}", e)
        return f"Ошибка при поиске: {e}"


@tool
def get_yclients_slots(service_name: str, date_from: str = "", date_to: str = "") -> str:
    """Получение свободных слотов для записи в YCLIENTS.

    Args:
        service_name: Название услуги (например, "Полная защита смартфона").
        date_from: Начальная дата в ISO формате (YYYY-MM-DD).
        date_to: Конечная дата в ISO формате (YYYY-MM-DD).

    Returns:
        Список доступных слотов в текстовом формате.
    """
    if _yclients_agent is None:
        return "Интеграция с YCLIENTS не настроена."

    try:
        import asyncio
        from datetime import date as dt

        d_from = dt.fromisoformat(date_from) if date_from else dt.today()
        d_to = dt.fromisoformat(date_to) if date_to else d_from.replace(
            day=d_from.day + 7
        )

        # Ищем ID услуги
        service_id = asyncio.get_event_loop().run_until_complete(
            _yclients_agent._find_service_id(service_name)
        )
        if service_id is None:
            return f"Услуга '{service_name}' не найдена в YCLIENTS."

        slots = asyncio.get_event_loop().run_until_complete(
            _yclients_agent.get_available_slots(
                service_id=service_id,
                date_from=d_from,
                date_to=d_to,
            )
        )

        if not slots:
            return "На ближайшие дни нет свободных слотов."

        result = ["Доступные слоты для записи:"]
        for slot in slots[:10]:
            date_str = slot.get("_date", "?")
            time_str = slot.get("time", slot.get("datetime", "?"))
            result.append(f"  📅 {date_str} в {time_str}")

        return "\n".join(result)
    except Exception as e:
        logger.error("Ошибка получения слотов YCLIENTS: {}", e)
        return f"Ошибка при получении слотов: {e}"


@tool
def create_yclients_booking(
    client_name: str,
    client_phone: str,
    service_name: str,
    datetime_str: str,
) -> str:
    """Создание записи клиента через YCLIENTS.

    Args:
        client_name: Имя клиента.
        client_phone: Телефон клиента.
        service_name: Название услуги.
        datetime_str: Дата и время в ISO формате (YYYY-MM-DDTHH:MM).

    Returns:
        Подтверждение записи или описание ошибки.
    """
    if _yclients_agent is None:
        return "Интеграция с YCLIENTS не настроена."

    try:
        import asyncio

        service_id = asyncio.get_event_loop().run_until_complete(
            _yclients_agent._find_service_id(service_name)
        )
        if service_id is None:
            return f"Услуга '{service_name}' не найдена в YCLIENTS."

        result = asyncio.get_event_loop().run_until_complete(
            _yclients_agent.book_appointment(
                client_name=client_name,
                client_phone=client_phone,
                service_id=service_id,
                appointment_datetime=datetime_str,
            )
        )

        if result:
            return (
                f"✅ Запись создана!\n"
                f"Клиент: {client_name}\n"
                f"Телефон: {client_phone}\n"
                f"Дата и время: {datetime_str}\n"
                f"Услуга: {service_name}"
            )
        return "Не удалось создать запись. Попробуйте позже или обратитесь к администратору."
    except Exception as e:
        logger.error("Ошибка создания записи в YCLIENTS: {}", e)
        return f"Ошибка при создании записи: {e}"
