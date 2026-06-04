"""RAG-агент: поиск релевантных фрагментов в базе знаний.

Выполняет поиск в векторной БД по запросу пользователя.
Возвращает релевантные чанки для использования в ответе.
"""

from typing import List
from langchain_core.documents import Document
from src.rag.vector_store import VectorStore
from src.utils.logger import logger


class RagAgent:
    """Агент поиска знаний в векторной БД."""

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self._vector_store = vector_store
        self._top_k = top_k

    def search(self, query: str, k: int | None = None) -> List[Document]:
        """Поиск релевантных документов по запросу."""
        k = k or self._top_k
        results = self._vector_store.search(query, k=k)
        logger.debug("RAG-агент: найдено {} документов по запросу: {}",
                      len(results),
                      query[:80])
        return results

    def format_context(self, documents: List[Document]) -> str:
        """Форматирование найденных чанков в строку контекста для LLM."""
        if not documents:
            return ""

        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("file_name", "неизвестно")
            parts.append(f"[Источник {i}: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)
