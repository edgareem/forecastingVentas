"""Funciones base para prediccion."""

import pandas as pd


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve predicciones placeholder para completar mas adelante."""
    result = df.copy()
    result["prediction"] = 0
    return result