"""Векторное хранилище: Chroma (dev) или Qdrant (prod).

Выбор бэкенда определяется переменной VECTOR_STORE_BACKEND в .env.
"""

from typing import List, Optional
from langchain_core.documents import Document
from src.utils.config import settings
from src.utils.logger import logger


class VectorStore:
    """Абстракция над векторной БД с поддержкой Chroma и Qdrant."""

    def __init__(self):
        self._store = None
        self._backend = settings.vector_store_backend

    def initialize(self):
        """Инициализация векторной БД в зависимости от выбранного бэкенда."""
        if self._backend == "chroma":
            self._init_chroma()
        elif self._backend == "qdrant":
            self._init_qdrant()
        else:
            raise ValueError(
                f"Неподдерживаемый backend: {self._backend}. "
                f"Допустимые значения: chroma, qdrant"
            )
        logger.info("Векторная БД инициализирована: backend={}", self._backend)

    def _init_chroma(self):
        """Инициализация локальной ChromaDB."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from langchain_chroma import Chroma

        self._chroma_client = chromadb.Client(ChromaSettings(
            is_persistent=True,
            persist_directory=settings.chroma_persist_dir,
        ))
        self._collection = self._chroma_client.get_or_create_collection(
            name="bronoskins_knowledge"
        )
        # LangChain-обёртка для удобного add_documents/search
        self._store = Chroma(
            client=self._chroma_client,
            collection_name="bronoskins_knowledge",
            embedding_function=self._get_langchain_embeddings(),
        )

    def _init_qdrant(self):
        """Инициализация Qdrant через Docker."""
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        self._qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._store = QdrantVectorStore(
            client=self._qdrant_client,
            collection_name="bronoskins_knowledge",
            embedding=self._get_langchain_embeddings(),
        )

    def _get_langchain_embeddings(self):
        """LangChain-совместимый wrapper для sentence-transformers."""
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Добавление документов в векторную БД."""
        if not documents:
            return []
        ids = self._store.add_documents(documents)
        logger.info("Добавлено {} документов в векторную БД", len(ids))
        return ids

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Поиск релевантных документов по запросу."""
        results = self._store.similarity_search(query, k=k)
        return results

    def clear(self):
        """Очистка коллекции для переиндексации."""
        if self._backend == "chroma":
            self._chroma_client.delete_collection("bronoskins_knowledge")
            self._collection = self._chroma_client.get_or_create_collection(
                name="bronoskins_knowledge"
            )
            from langchain_chroma import Chroma
            self._store = Chroma(
                client=self._chroma_client,
                collection_name="bronoskins_knowledge",
                embedding_function=self._get_langchain_embeddings(),
            )
        elif self._backend == "qdrant":
            self._qdrant_client.delete_collection("bronoskins_knowledge")
        logger.info("Коллекция очищена для переиндексации")
