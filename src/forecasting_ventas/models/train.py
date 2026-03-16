"""Punto de entrada inicial para entrenamiento."""

from pathlib import Path


def train(output_path: str | Path | None = None) -> None:
    """Funcion base para implementar el entrenamiento del modelo."""
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)