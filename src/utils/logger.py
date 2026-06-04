"""Настройка логирования через loguru с ротацией и форматированием."""

import sys
from loguru import logger as _logger
from src.utils.config import settings


def setup_logger():
    """Инициализация логгера с выводом в консоль и файл с ротацией."""
    # Удаляем стандартный обработчик
    _logger.remove()

    # Консольный вывод
    _logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # Файловый вывод с ротацией
    _logger.add(
        "logs/bronoskins_bot_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="gz",
        encoding="utf-8",
    )

    return _logger


logger = setup_logger()
