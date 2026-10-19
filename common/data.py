"""Carga y preparación del dataset de ventas, compartida por las tareas B y C (responsable: Elizabeth).

El dataset tiene solo 3 categorías originales (Beauty, Clothing, Electronics). Con autorización del
profesor, la Tarea B trabaja con 6 series que resultan de cruzar Categoría × Género.
"""
from pathlib import Path

import pandas as pd

from common import config

# Columnas que debe traer el CSV y el tipo con el que se trabajan.
EXPECTED_COLUMNS = {
    "Transaction ID": "int64",
    "Date": "datetime64[ns]",
    "Customer ID": "object",
    "Gender": "object",
    "Age": "int64",
    "Product Category": "object",
    "Quantity": "int64",
    "Price per Unit": "int64",
    "Total Amount": "int64",
}

CATEGORIES = ["Clothing", "Beauty", "Electronics"]  # las 3 categorías originales
GENDERS = ["Male", "Female"]
# Nombres exactos de las 6 series (Categoría-Género), en orden fijo.
SERIES_NAMES = [f"{c}-{g}" for c in CATEGORIES for g in GENDERS]

AGE_BINS = [17, 25, 35, 45, 55, 65]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-64"]

# Una semana empieza en domingo (el CSV empieza el domingo 2023-01-01).
WEEK_START = "SUN"
_WEEKDAY_INDEX = {"MON": 0, "SUN": 6}  # convención de pandas: lunes = 0

TARGETS = {"units": "Quantity", "revenue": "Total Amount"}


def load_sales(path: Path | str = config.SALES_CSV) -> pd.DataFrame:
    """Lee el CSV de ventas (sin modificarlo), convierte Date a fecha y valida columnas y tipos.

    Lanza ValueError si faltan columnas, hay nulos o Total Amount no es Quantity × Price per Unit.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")
    df = df[list(EXPECTED_COLUMNS)].astype(EXPECTED_COLUMNS)
    if df.isna().any().any():
        raise ValueError("El CSV tiene valores nulos")
    if not (df["Quantity"] * df["Price per Unit"] == df["Total Amount"]).all():
        raise ValueError("Total Amount no coincide con Quantity × Price per Unit")
    unknown = set(df["Product Category"]) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Categorías inesperadas: {sorted(unknown)}")
    return df


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con dos columnas nuevas:

    - "Series": "Categoría-Género", p. ej. "Clothing-Female" (6 valores posibles).
    - "Age Group": rango de edad ("18-25", ..., "56-64").
    """
    out = df.copy()
    out["Series"] = out["Product Category"] + "-" + out["Gender"]
    out["Age Group"] = pd.cut(out["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return out


def week_start_of(dates: pd.Series, week_start: str = WEEK_START) -> pd.Series:
    """Fecha (a medianoche) del primer día de la semana a la que pertenece cada fecha."""
    first_day = _WEEKDAY_INDEX[week_start]
    days_back = (dates.dt.dayofweek - first_day) % 7
    return dates.dt.normalize() - pd.to_timedelta(days_back, unit="D")


def weekly_series(df: pd.DataFrame, target: str = "units", week_start: str = WEEK_START) -> pd.DataFrame:
    """Ventas semanales por serie en formato ancho.

    - Índice: fecha de inicio de cada semana (nombre "week_start").
    - Columnas: las 6 series de SERIES_NAMES, en ese orden.
    - target="units" suma Quantity; target="revenue" suma Total Amount.
    - Solo se conservan semanas COMPLETAS (7 días dentro del rango de fechas del df); los días de
      semanas incompletas en los extremos se descartan.
    - Una semana sin transacciones de una serie vale 0 (hipótesis: no hubo ventas esa semana).
    """
    if target not in TARGETS:
        raise ValueError(f"target debe ser uno de {list(TARGETS)}")
    if "Series" not in df.columns:
        df = add_segments(df)

    weeks = week_start_of(df["Date"], week_start)
    one_week = pd.Timedelta(7, unit="D")
    # Primera semana completa: si el df no empieza justo el primer día de su semana, se salta esa semana.
    first_full = weeks.min()
    if df["Date"].min().normalize() > first_full:
        first_full += one_week
    # Última semana completa: si el df termina antes del último día de su semana, se descarta.
    last_full = weeks.max()
    if df["Date"].max().normalize() < last_full + pd.Timedelta(6, unit="D"):
        last_full -= one_week

    all_weeks = pd.date_range(first_full, last_full, freq="7D", name="week_start")
    keep = (weeks >= first_full) & (weeks <= last_full)
    table = (
        df[keep]
        .assign(week_start=weeks[keep])
        .pivot_table(index="week_start", columns="Series", values=TARGETS[target], aggfunc="sum")
    )
    # reindex rellena con 0 las semanas o series sin ventas.
    table = table.reindex(index=all_weeks, columns=SERIES_NAMES, fill_value=0).fillna(0)
    table.columns.name = None
    return table.astype("int64")


def pricing_stats(df: pd.DataFrame, by: tuple[str, ...] = ("Product Category", "Gender")) -> pd.DataFrame:
    """Escalas de precio y compra por segmento (pensada para la Tarea C).

    Firma estable usada por la Tarea C: cambiarla requiere pull request revisado por ambos.

    Parámetros:
      df: transacciones de load_sales() (con o sin las columnas de add_segments).
      by: columnas que definen el segmento, p. ej. ("Product Category", "Gender"), ("Series",)
          o ("Age Group",).
    Devuelve un DataFrame con una fila por segmento: las columnas de `by` y
      n_transactions, units, revenue, price_min, price_mean, price_median, price_max,
      qty_mean, age_mean.
    Ojo: el dataset solo registra compras; esto da escalas reales, no la reacción al precio.
    """
    if "Series" not in df.columns:
        df = add_segments(df)
    grouped = df.groupby(list(by), observed=True)
    stats = grouped.agg(
        n_transactions=("Transaction ID", "count"),
        units=("Quantity", "sum"),
        revenue=("Total Amount", "sum"),
        price_min=("Price per Unit", "min"),
        price_mean=("Price per Unit", "mean"),
        price_median=("Price per Unit", "median"),
        price_max=("Price per Unit", "max"),
        qty_mean=("Quantity", "mean"),
        age_mean=("Age", "mean"),
    )
    return stats.reset_index()
