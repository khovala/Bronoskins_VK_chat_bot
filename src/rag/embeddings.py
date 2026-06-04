"""Инициализация модели эмбеддингов.

Используется intfloat/multilingual-e5-large — одна из лучших
мультиязычных моделей для русского языка.
При первом запуске модель скачивается автоматически (~2 ГБ).
"""

from typing import List
from sentence_transformers import SentenceTransformer
from src.utils.config import settings
from src.utils.logger import logger


class EmbeddingsManager:
    """Менеджер эмбеддингов на основе sentence-transformers."""

    def __init__(self):
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Ленивая инициализация модели."""
        if self._model is None:
            logger.info("Загрузка модели эмбеддингов: {}...",
                         settings.embedding_model)
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info("Модель эмбеддингов загружена успешно")
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для списка текстов."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Генерация эмбеддинга для одного запроса."""
        embedding = self.model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()


# Глобальный экземпляр для переиспользования между модулями
embeddings_manager = EmbeddingsManager()
