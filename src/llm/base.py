"""Абстрактный базовый класс для LLM-клиентов.

Позволяет легко переключаться между YandexGPT и другими провайдерами.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Базовый интерфейс для всех LLM-провайдеров."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Генерация ответа LLM.

        Args:
            prompt: Запрос пользователя.
            system_prompt: Системный промпт для настройки поведения.
            temperature: Креативность ответа (0.0–1.0).
            max_tokens: Максимальная длина ответа.

        Returns:
            Сгенерированный текст ответа.
        """
        ...
