"""Configuration-driven extractors for flexible web scraping"""

from .base import AbstractExtractor, ExtractionResult
from .config_based import ConfigBasedExtractor

__all__ = [
    'AbstractExtractor',
    'ExtractionResult',
    'ConfigBasedExtractor',
]
