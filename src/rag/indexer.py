"""Скрипт индексации документов в векторную БД.

Запуск: python -m src.rag.indexer
При запуске сканирует папку data/, загружает документы,
разбивает на чанки, генерирует эмбеддинги и сохраняет в БД.
Поддерживает флаг --clear для переиндексации без дублирования.
"""

import argparse
from pathlib import Path
from src.rag.loader import load_documents
from src.rag.splitter import split_documents
from src.rag.vector_store import VectorStore
from src.utils.config import settings
from src.utils.logger import logger


def run_indexing(clear: bool = False):
    """Основной пайплайн индексации."""
    logger.info("=" * 60)
    logger.info("Запуск индексации документов Bronoskins")
    logger.info("=" * 60)

    # 1. Загрузка документов
    data_dir = Path(settings.documents_path)
    logger.info("Сканирование папки: {}", data_dir)
    documents = load_documents(data_dir)

    if not documents:
        logger.warning("Документы не найдены. Проверьте содержимое папки data/")
        return

    # 2. Разбивка на чанки
    logger.info("Разбивка {} документов на чанки...", len(documents))
    chunks = split_documents(documents)
    logger.info("Получено {} чанков", len(chunks))

    # 3. Сохранение в векторную БД
    vector_store = VectorStore()
    vector_store.initialize()

    if clear:
        logger.info("Очистка существующей коллекции...")
        vector_store.clear()

    vector_store.add_documents(chunks)
    logger.info("=" * 60)
    logger.info("Индексация завершена успешно!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Индексация документов Bronoskins")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Очистить коллекцию перед индексацией (переиндексация без дублирования)",
    )
    args = parser.parse_args()
    run_indexing(clear=args.clear)
