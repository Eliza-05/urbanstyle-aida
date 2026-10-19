"""Carga y preparación de datos de ventas (responsable: Elizabeth)."""
import pandas as pd


def load_sales() -> pd.DataFrame:
    """Lee data/raw/retail_sales_dataset.csv sin modificarlo."""
    raise NotImplementedError


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de segmentación (p. ej. edad, género, categoría)."""
    raise NotImplementedError


def weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las ventas en series semanales."""
    raise NotImplementedError


def pricing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas de precio/cantidad por producto o categoría."""
    raise NotImplementedError
