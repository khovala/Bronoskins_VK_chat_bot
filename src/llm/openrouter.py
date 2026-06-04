"""Опциональный клиент для OpenRouter API.

Используется для тестирования, когда YandexGPT недоступен.
Требует OPENROUTER_API_KEY и OPENROUTER_MODEL в .env.
"""

from typing import Optional
import aiohttp
from src.llm.base import BaseLLM
from src.utils.config import settings
from src.utils.logger import logger

try:
    OPENROUTER_API_KEY = settings.openrouter_api_key
    OPENROUTER_MODEL = settings.openrouter_model
except AttributeError:
    OPENROUTER_API_KEY = ""
    OPENROUTER_MODEL = "google/gemini-flash-1.5"


class OpenRouterClient(BaseLLM):
    """Клиент для OpenRouter API — мультипровайдерный доступ к LLM."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._api_key = api_key or OPENROUTER_API_KEY
        self._model = model or OPENROUTER_MODEL

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY не задан в .env"
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bronoskins.ru",
            "X-Title": "Bronoskins Bot",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"OpenRouter API error {response.status}: {error_text[:200]}"
                    )

                data = await response.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    logger.debug("OpenRouter ответ: {} символов", len(content))
                    return content

                raise RuntimeError("OpenRouter: пустой ответ")
