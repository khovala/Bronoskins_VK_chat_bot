"""Загрузчики документов для RAG-пайплайна.

Поддерживаются форматы: .docx, .pdf, .txt, .md.
Для .docx используется docx2txt, для .pdf — PyPDFLoader из LangChain.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.utils.logger import logger


def load_docx(file_path: Path) -> List[Document]:
    """Загрузка .docx файла с помощью docx2txt."""
    import docx2txt
    text = docx2txt.process(str(file_path))
    if not text or not text.strip():
        logger.warning("Пустой документ: {}", file_path)
        return []
    metadata = {"source": str(file_path), "file_type": "docx",
                 "file_name": file_path.name}
    return [Document(page_content=text, metadata=metadata)]


def load_pdf(file_path: Path) -> List[Document]:
    """Загрузка .pdf через PyPDFLoader."""
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["file_type"] = "pdf"
        doc.metadata["file_name"] = file_path.name
    logger.info("Загружен PDF: {} ({} страниц)", file_path.name, len(docs))
    return docs


def load_text(file_path: Path) -> List[Document]:
    """Загрузка текстовых файлов (.txt, .md)."""
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["file_type"] = file_path.suffix.lstrip(".")
        doc.metadata["file_name"] = file_path.name
    return docs


LOADER_MAP = {
    ".docx": load_docx,
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
}


def load_documents(data_dir: Path) -> List[Document]:
    """Рекурсивный обход папки data/ и загрузка всех поддерживаемых документов."""
    all_docs: List[Document] = []
    if not data_dir.exists():
        logger.error("Папка с данными не найдена: {}", data_dir)
        return all_docs

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        loader = LOADER_MAP.get(ext)
        if loader is None:
            logger.debug("Пропущен неподдерживаемый формат: {}", file_path)
            continue
        try:
            docs = loader(file_path)
            all_docs.extend(docs)
            logger.info("Загружен документ: {} (чанков: {})",
                         file_path.name, len(docs))
        except Exception as e:
            logger.error("Ошибка загрузки {}: {}", file_path, e)

    logger.info("Всего загружено документов: {}, чанков: {}",
                 len({d.metadata.get("file_name") for d in all_docs}),
                 len(all_docs))
    return all_docs
