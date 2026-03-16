"""Utilidades basicas para cargar datos tabulares."""

from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """Carga un archivo CSV en un DataFrame de pandas."""
    return pd.read_csv(file_path, **kwargs)