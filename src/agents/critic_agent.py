"""Агент-Критик: оценка релевантности найденных RAG-фрагментов.

Вычисляет косинусное сходство между эмбеддингом запроса и эмбеддингами чанков.
Если сходство ниже порога — генерирует улучшенный запрос через LLM
и инициирует повторный поиск (до 2 попыток).
"""

from typing import List, Tuple
import numpy as np
from langchain_core.documents import Document
from src.rag.embeddings import embeddings_manager
from src.llm.base import BaseLLM
from src.utils.config import settings
from src.utils.logger import logger


class CriticAgent:
    """Оценщик релевантности результатов RAG-поиска."""

    def __init__(self, llm_client: BaseLLM | None = None):
        self._llm = llm_client
        self._threshold = settings.relevance_threshold

    def evaluate(
        self, query: str, documents: List[Document]
    ) -> Tuple[bool, float]:
        """Оценка релевантности найденных документов.

        Returns:
            Tuple[bool, float]: (релевантны ли, максимальное сходство)
        """
        if not documents:
            logger.debug("Критик: документы не найдены")
            return False, 0.0

        # Получаем эмбеддинг запроса
        query_embedding = np.array(embeddings_manager.embed_query(query))

        # Получаем эмбеддинги чанков
        chunk_texts = [doc.page_content for doc in documents]
        chunk_embeddings = np.array(embeddings_manager.embed_texts(chunk_texts))

        # Вычисляем косинусное сходство
        similarities = self._cosine_similarity(query_embedding, chunk_embeddings)
        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))

        logger.debug(
            "Критик: макс. схожесть={:.3f}, средняя={:.3f}, порог={:.3f}",
            max_similarity,
            avg_similarity,
            self._threshold,
        )

        return max_similarity >= self._threshold, max_similarity

    def generate_refined_query(
        self, original_query: str, documents: List[Document]
    ) -> str | None:
        """Генерация уточнённого запроса через LLM для улучшения поиска."""
        if self._llm is None:
            logger.warning(
                "Критик: LLM не задан, "
                "невозможно сгенерировать уточнённый запрос"
            )
            return None

        context = "\n".join(
            [d.page_content[:300] for d in documents[:2]]
        )
        prompt = (
            f"Пользователь спросил: «{original_query}»\n\n"
            f"База знаний вернула:\n{context}\n\n"
            f"Результаты не очень релевантны. "
            f"Переформулируй запрос пользователя так, "
            f"чтобы улучшить поиск в базе знаний компании "
            f"по защитным плёнкам Bronoskins. "
            f"Верни ТОЛЬКО переформулированный запрос, без пояснений."
        )
        try:
            import asyncio
            refined = asyncio.get_event_loop().run_until_complete(
                self._llm.generate(prompt, temperature=0.3, max_tokens=200)
            )
            logger.info("Критик: уточнённый запрос: {}", refined[:100])
            return refined.strip()
        except Exception as e:
            logger.error("Критик: ошибка генерации уточнённого запроса: {}", e)
            return None

    @staticmethod
    def _cosine_similarity(
        query_vec: np.ndarray, chunk_vecs: np.ndarray
    ) -> np.ndarray:
        """Вычисление косинусного сходства между запросом и чанками."""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        chunk_norms = chunk_vecs / (
            np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-10
        )
        return np.dot(chunk_norms, query_norm)
