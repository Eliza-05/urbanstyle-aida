"""Pruebas de common/data.py: unitarias con datos sintéticos (sin CSV) y una de integración con el CSV real."""
import pandas as pd
import pytest

from common import config
from common.data import (
    SERIES_NAMES,
    add_segments,
    load_sales,
    pricing_stats,
    week_start_of,
    weekly_series,
)


def make_sales(rows):
    """DataFrame con el mismo formato del CSV a partir de (fecha, categoría, género, edad, cantidad, precio)."""
    records = []
    for i, (date, cat, gender, age, qty, price) in enumerate(rows, start=1):
        records.append({
            "Transaction ID": i, "Date": pd.Timestamp(date), "Customer ID": f"C{i}",
            "Gender": gender, "Age": age, "Product Category": cat,
            "Quantity": qty, "Price per Unit": price, "Total Amount": qty * price,
        })
    return pd.DataFrame(records)


# Domingo 2023-01-01 a sábado 2023-01-21 = 3 semanas completas Dom-Sáb, más un domingo suelto al final.
SYNTH = make_sales([
    ("2023-01-01", "Clothing", "Female", 20, 2, 50),     # semana 1 (domingo)
    ("2023-01-07", "Clothing", "Female", 30, 1, 50),     # semana 1 (sábado)
    ("2023-01-09", "Beauty", "Male", 40, 3, 25),         # semana 2
    ("2023-01-21", "Electronics", "Male", 64, 4, 300),   # semana 3 (sábado)
    ("2023-01-22", "Electronics", "Female", 18, 1, 500), # semana 4 incompleta -> se descarta
])


def test_add_segments_creates_series_and_age_groups():
    seg = add_segments(SYNTH)
    assert list(seg["Series"]) == [
        "Clothing-Female", "Clothing-Female", "Beauty-Male", "Electronics-Male", "Electronics-Female"]
    assert list(seg["Age Group"]) == ["18-25", "26-35", "36-45", "56-64", "18-25"]
    assert "Series" not in SYNTH.columns  # no modifica el original


def test_week_start_sunday_and_monday():
    dates = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-07", "2023-01-08", "2023-01-02"]))
    assert list(week_start_of(dates, "SUN").dt.day) == [1, 1, 8, 1]
    assert list(week_start_of(dates, "MON").dt.day) == [26, 2, 2, 2]  # 2023-01-01 es domingo -> semana del 26-dic


def test_weekly_series_shape_zeros_and_totals():
    weekly = weekly_series(SYNTH, target="units")
    assert list(weekly.columns) == SERIES_NAMES
    assert list(weekly.index) == list(pd.date_range("2023-01-01", periods=3, freq="7D"))
    assert weekly.loc["2023-01-01", "Clothing-Female"] == 3
    assert weekly.loc["2023-01-08", "Beauty-Male"] == 3
    assert weekly.loc["2023-01-15", "Electronics-Male"] == 4
    assert weekly.loc["2023-01-08", "Clothing-Female"] == 0  # semana sin ventas -> 0
    # La suma semanal reconstruye exactamente el total de las filas dentro de las semanas completas.
    assert weekly.to_numpy().sum() == SYNTH["Quantity"].iloc[:4].sum()


def test_weekly_series_revenue():
    weekly = weekly_series(SYNTH, target="revenue")
    assert weekly.to_numpy().sum() == SYNTH["Total Amount"].iloc[:4].sum()
    with pytest.raises(ValueError):
        weekly_series(SYNTH, target="profit")


def test_pricing_stats_per_segment():
    stats = pricing_stats(SYNTH)
    row = stats[(stats["Product Category"] == "Clothing") & (stats["Gender"] == "Female")].iloc[0]
    assert row["n_transactions"] == 2
    assert row["units"] == 3
    assert row["revenue"] == 150
    assert row["price_min"] == 50
    assert row["price_max"] == 50
    by_age = pricing_stats(SYNTH, by=("Age Group",))
    assert by_age["n_transactions"].sum() == len(SYNTH)


def test_load_sales_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    SYNTH.drop(columns=["Gender"]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_sales(bad)


@pytest.mark.skipif(not config.SALES_CSV.exists(), reason="CSV real no disponible")
def test_integration_real_csv():
    df = load_sales()
    assert len(df) == 1000
    weekly = weekly_series(df, target="units")
    assert weekly.shape == (52, 6)
    assert weekly.index[0] == pd.Timestamp("2023-01-01")
    assert weekly.index[-1] == pd.Timestamp("2023-12-24")
    # Solo se descartan las filas de los días fuera de las 52 semanas completas (31-dic y 1-ene).
    inside = df["Date"] < pd.Timestamp("2023-12-31")
    assert weekly.to_numpy().sum() == df.loc[inside, "Quantity"].sum()
    assert (~inside).sum() == 3
