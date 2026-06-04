"""Конфигурация приложения через переменные окружения (.env).

Все секреты и настройки вынесены в .env — никакого хардкода.
Для переопределения параметров используются переменные окружения.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из .env и переменных окружения."""

    # VK
    vk_token: str = ""

    # YandexGPT
    yandexgpt_api_key: str = ""
    yandexgpt_folder_id: str = ""
    yandexgpt_model: str = "yandexgpt-lite"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Chroma (dev)
    chroma_persist_dir: str = "./chroma_db"

    # YCLIENTS
    yclients_token: str = ""
    # JSON-строка: {"Название филиала": company_id, ...}
    yclients_filials: str = "{}"

    # Режим работы векторной БД: "chroma" или "qdrant"
    vector_store_backend: str = "chroma"

    # Эмбеддинги
    embedding_model: str = "intfloat/multilingual-e5-large"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_top_k: int = 5
    relevance_threshold: float = 0.7

    # История диалога
    max_history_messages: int = 10

    # Loguru
    log_level: str = "INFO"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"

    # Путь к данным
    data_dir: Path = Path("data")

    # Адреса торговых центров Bronoskins (Тула)
    # Можно переопределить через .env как JSON-строку: YCLIENTS_LOCATIONS='{"Макси":"...","Гостиный двор":"..."}'
    yclients_locations: str = (
        '{"Макси": "ТРЦ Макси, ул. Пролетарская, д. 2",'
        ' "Гостиный двор": "Гостиный двор, ул. Советская, д. 47",'
        ' "Первый": "ТЦ Первый, ул. Октябрьская, д. 1"}'
    )

    # Путь к папке с исходными документами
    documents_path: str = "data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Глобальный экземпляр настроек (инициализируется при старте приложения)
settings = Settings()
