"""Агент интеграции с YCLIENTS API — поддержка нескольких филиалов.

Каждый филиал имеет свой company_id в YCLIENTS.
Агент умеет искать слоты и создавать записи в любом из филиалов.
Все вызовы асинхронные с обработкой ошибок сети и таймаутов.
"""

import json
import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List

import aiohttp
from src.utils.config import settings
from src.utils.logger import logger


class YClientsAgent:
    """Агент для взаимодействия с YCLIENTS API (мульти-филиальный)."""

    BASE_URL = "https://api.yclients.com/api/v1"

    def __init__(
        self,
        token: Optional[str] = None,
        filials: Optional[Dict[str, int]] = None,
    ):
        self._token = token or settings.yclients_token
        self._filials = filials or self._parse_filials()
        self._locations = self._parse_locations()

    @staticmethod
    def _parse_filials() -> Dict[str, int]:
        """Парсинг маппинга филиалов из YCLIENTS_FILIALS."""
        try:
            raw = json.loads(settings.yclients_filials)
            result = {str(k): int(v) for k, v in raw.items()}
            logger.info(
                "YCLIENTS: загружено {} филиалов: {}",
                len(result),
                list(result.keys()),
            )
            return result
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning(
                "YCLIENTS: не удалось распарсить YCLIENTS_FILIALS ({}), "
                "филиалы не настроены",
                e,
            )
            return {}

    @staticmethod
    def _parse_locations() -> Dict[str, str]:
        """Парсинг адресов торговых центров из конфигурации."""
        try:
            return json.loads(settings.yclients_locations)
        except (json.JSONDecodeError, AttributeError):
            return {
                "Макси": "ТРЦ Макси, ул. Пролетарская, д. 2",
                "Гостиный двор": "Гостиный двор, ул. Советская, д. 47",
                "Новомосковск": "ТЦ Новомосковск",
            }

    @property
    def filial_names(self) -> List[str]:
        """Список названий всех настроенных филиалов."""
        return list(self._filials.keys())

    def get_company_id(self, filial_name: str) -> int | None:
        """Получить company_id по названию филиала (нечёткий поиск)."""
        # Прямое совпадение
        if filial_name in self._filials:
            return self._filials[filial_name]
        # Нечёткий поиск
        name_lower = filial_name.lower()
        for fname, cid in self._filials.items():
            if name_lower in fname.lower() or fname.lower() in name_lower:
                return cid
        return None

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _request(
        self, method: str, path: str, **kwargs
    ) -> dict | list | None:
        """Базовый метод для запросов к API YCLIENTS с retry."""
        url = f"{self.BASE_URL}{path}"
        timeout = aiohttp.ClientTimeout(total=30)
        last_error = None

        for attempt in range(1, 4):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=url,
                        headers=self._headers,
                        timeout=timeout,
                        **kwargs,
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get("data", data)

                        if response.status >= 500 or response.status == 429:
                            wait = 2 ** attempt
                            logger.warning(
                                "YCLIENTS: ошибка {}, "
                                "повтор через {} сек, попытка {}/3",
                                response.status, wait, attempt,
                            )
                            await asyncio.sleep(wait)
                            continue

                        error_text = await response.text()
                        logger.error(
                            "YCLIENTS: ошибка {} — {}",
                            response.status,
                            error_text[:200],
                        )
                        return None

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "YCLIENTS: сетевая ошибка, "
                    "повтор через {} сек, попытка {}/3",
                    wait, attempt,
                )
                await asyncio.sleep(wait)

        logger.error(
            "YCLIENTS: все попытки исчерпаны. "
            "Последняя ошибка: {}", last_error,
        )
        return None

    async def get_services(self, company_id: int) -> List[dict]:
        """Получение списка услуг конкретного филиала."""
        result = await self._request(
            "GET", f"/company/{company_id}/services"
        )
        if result and isinstance(result, list):
            return result
        if result and isinstance(result, dict):
            return result.get("services", [])
        return []

    async def _find_service_id(
        self, service_name: str, company_id: int
    ) -> int | None:
        """Поиск ID услуги по названию в конкретном филиале."""
        services = await self.get_services(company_id)
        service_lower = service_name.lower()

        for svc in services:
            title = svc.get("title", "").lower()
            if service_lower in title or title in service_lower:
                svc_id = svc.get("id")
                logger.info(
                    "YCLIENTS [{}]: найдена услуга '{}' (id={})",
                    company_id, svc.get("title"), svc_id,
                )
                return svc_id

        logger.warning(
            "YCLIENTS [{}]: услуга '{}' не найдена среди {} услуг",
            company_id, service_name, len(services),
        )
        return None

    async def get_staff(self, company_id: int) -> List[dict]:
        """Получение списка сотрудников филиала."""
        result = await self._request(
            "GET", f"/company/{company_id}/staff"
        )
        if result and isinstance(result, list):
            return result
        if result and isinstance(result, dict):
            return result.get("staff", [])
        return []

    async def get_available_slots(
        self,
        company_id: int,
        service_id: int | None = None,
        staff_id: int | None = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> List[dict]:
        """Получение свободных слотов в конкретном филиале."""
        if date_from is None:
            date_from = date.today()
        if date_to is None:
            date_to = date_from + timedelta(days=7)

        if staff_id is None:
            staff = await self.get_staff(company_id)
            if staff:
                staff_id = staff[0].get("id")
            else:
                logger.warning(
                    "YCLIENTS [{}]: нет доступных сотрудников", company_id
                )
                return []

        if staff_id is None:
            return []

        slots: List[dict] = []
        current_date = date_from
        while current_date <= date_to:
            payload = {
                "service_ids": [service_id] if service_id else [],
                "staff_id": staff_id,
                "date": current_date.isoformat(),
            }
            result = await self._request(
                "POST",
                f"/company/{company_id}/book_staff",
                json=payload,
            )
            if result:
                day_slots = (
                    result if isinstance(result, list)
                    else result.get("slots", [])
                )
                for slot in day_slots:
                    slot["_date"] = current_date.isoformat()
                    slot["_company_id"] = company_id
                slots.extend(day_slots)
            current_date += timedelta(days=1)

        logger.info(
            "YCLIENTS [{}]: найдено {} слотов ({} — {})",
            company_id, len(slots), date_from, date_to,
        )
        return slots

    async def search_all_filials_for_slots(
        self,
        service_name: str,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> Dict[str, List[dict]]:
        """Поиск свободных слотов во ВСЕХ филиалах.

        Returns:
            Dict[название_филиала, список_слотов]
        """
        if date_from is None:
            date_from = date.today()
        if date_to is None:
            date_to = date_from + timedelta(days=7)

        all_results: Dict[str, List[dict]] = {}

        for filial_name, company_id in self._filials.items():
            service_id = await self._find_service_id(service_name, company_id)
            if service_id is None:
                logger.warning(
                    "YCLIENTS: услуга '{}' не найдена в филиале '{}'",
                    service_name, filial_name,
                )
                all_results[filial_name] = []
                continue

            slots = await self.get_available_slots(
                company_id=company_id,
                service_id=service_id,
                date_from=date_from,
                date_to=date_to,
            )
            all_results[filial_name] = slots

        total = sum(len(s) for s in all_results.values())
        logger.info(
            "YCLIENTS: всего {} слотов по {} филиалам для услуги '{}'",
            total, len(self._filials), service_name,
        )
        return all_results

    async def book_appointment(
        self,
        company_id: int,
        client_name: str,
        client_phone: str,
        service_id: int,
        appointment_datetime: str,
        staff_id: Optional[int] = None,
    ) -> dict | None:
        """Создание записи клиента в конкретном филиале."""
        if staff_id is None:
            staff = await self.get_staff(company_id)
            if staff:
                staff_id = staff[0].get("id")

        if staff_id is None:
            logger.error(
                "YCLIENTS [{}]: не удалось определить сотрудника", company_id
            )
            return None

        payload = {
            "client": {
                "name": client_name,
                "phone": client_phone,
            },
            "service_id": service_id,
            "staff_id": staff_id,
            "datetime": appointment_datetime,
        }

        result = await self._request(
            "POST",
            f"/company/{company_id}/book_record",
            json=payload,
        )

        if result:
            logger.info(
                "YCLIENTS [{}]: запись создана — клиент: {}, "
                "дата: {}, услуга: {}",
                company_id, client_name, appointment_datetime, service_id,
            )
        else:
            logger.error(
                "YCLIENTS [{}]: не удалось создать запись", company_id
            )

        return result if isinstance(result, dict) else None

    def get_locations_text(self) -> str:
        """Форматированный список адресов всех ТЦ."""
        locs = [
            f"📍 {name}: {addr}"
            for name, addr in self._locations.items()
        ]
        return "\n".join(locs)

    def get_location(self, name: str) -> str | None:
        """Получить адрес конкретного ТЦ по названию."""
        for loc_name, addr in self._locations.items():
            if name.lower() in loc_name.lower() or loc_name.lower() in name.lower():
                return addr
        return None

    def is_configured(self) -> bool:
        """Проверка, что хотя бы один филиал настроен."""
        return bool(self._token and self._filials)
