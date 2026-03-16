"""Helpers genericos reutilizables."""

from datetime import datetime


def timestamp() -> str:
    """Devuelve una marca temporal simple para nombres de archivos."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")