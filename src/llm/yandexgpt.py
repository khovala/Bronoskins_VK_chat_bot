"""Клиент для YandexGPT API.

Прямое обращение к API Yandex Cloud.
Эндпоинт: https://llm.api.cloud.yandex.net/foundationModels/v1/completion

Поддерживает автоматические повторные попытки (retry) при:
- Ошибках сети (timeout, connection)
- Ошибках сервера (5xx)
- Превышении лимита запросов (429)
"""

import asyncio
import json
from typing import Optional

import aiohttp
from src.llm.base import BaseLLM
from src.utils.config import settings
from src.utils.logger import logger


class YandexGPTClient(BaseLLM):
    """Асинхронный клиент YandexGPT с retry-логикой."""

    API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(
        self,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._api_key = api_key or settings.yandexgpt_api_key
        self._folder_id = folder_id or settings.yandexgpt_folder_id
        self._model = model or settings.yandexgpt_model

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Api-Key {self._api_key}",
            "x-folder-id": self._folder_id,
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Генерация ответа с retry при ошибках.

        До 3 попыток с экспоненциальной задержкой при 5xx и 429.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "text": system_prompt})
        messages.append({"role": "user", "text": prompt})

        payload = {
            "modelUri": f"gpt://{self._folder_id}/{self._model}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": messages,
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, 4):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.API_URL,
                        headers=self._headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = data.get("result", {})
                            alternatives = result.get("alternatives", [])
                            if alternatives:
                                text = alternatives[0].get("message", {}).get("text", "")
                                logger.debug(
                                    "YandexGPT ответ (попытка {}): {} символов",
                                    attempt,
                                    len(text),
                                )
                                return text
                            logger.warning("YandexGPT: пустой ответ, попытка {}", attempt)
                            continue

                        if response.status == 429:
                            retry_after = int(
                                response.headers.get("Retry-After", "5")
                            )
                            logger.warning(
                                "YandexGPT: rate limit (429), "
                                "ожидание {} сек, попытка {}/3",
                                retry_after,
                                attempt,
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        if response.status >= 500:
                            wait = 2 ** attempt
                            logger.warning(
                                "YandexGPT: ошибка сервера {}, "
                                "повтор через {} сек, попытка {}/3",
                                response.status,
                                wait,
                                attempt,
                            )
                            await asyncio.sleep(wait)
                            continue

                        # 4xx ошибки (кроме 429) — не повторяем
                        error_text = await response.text()
                        logger.error(
                            "YandexGPT: ошибка {} — {}",
                            response.status,
                            error_text[:200],
                        )
                        raise RuntimeError(
                            f"YandexGPT API error {response.status}: {error_text[:200]}"
                        )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "YandexGPT: сетевая ошибка, "
                    "повтор через {} сек, попытка {}/3: {}",
                    wait,
                    attempt,
                    e,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"YandexGPT: все 3 попытки исчерпаны. "
            f"Последняя ошибка: {last_error}"
        )
