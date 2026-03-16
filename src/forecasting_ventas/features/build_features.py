"""Transformaciones iniciales para preparar variables."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia del dataset lista para extender con nuevas features."""
    return df.copy()