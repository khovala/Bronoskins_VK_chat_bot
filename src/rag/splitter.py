"""Разбивка документов на чанки с перекрытием."""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config import settings


def create_splitter() -> RecursiveCharacterTextSplitter:
    """Создание сплиттера с настройками из конфигурации.

    separators подобраны для русского текста: сначала двойной перенос строки,
    затем одиночный, затем точка с пробелом, пробел, и наконец посимвольно.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(documents: List[Document]) -> List[Document]:
    """Разбивает список документов на чанки."""
    splitter = create_splitter()
    chunks = splitter.split_documents(documents)
    return chunks
