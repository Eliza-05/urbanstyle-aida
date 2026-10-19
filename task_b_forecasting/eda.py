"""Tarea B (Elizabeth): análisis exploratorio de las 6 series semanales; guarda gráficas y tabla en outputs/task_b/.

Uso: python -m task_b_forecasting.eda
"""
import matplotlib

matplotlib.use("Agg")  # sin ventana: solo se guardan archivos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import config
from common.data import SERIES_NAMES, load_sales, weekly_series

OUT_DIR = config.OUTPUTS_DIR / "task_b"
MAX_LAG = 4


def series_stats(weekly: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas por serie: nivel, dispersión, semanas en cero y autocorrelación."""
    rows = []
    for name in weekly.columns:
        s = weekly[name]
        row = {
            "series": name,
            "weeks": len(s),
            "total_units": int(s.sum()),
            "mean": s.mean(),
            "std": s.std(),
            "cv": s.std() / s.mean(),
            "min": int(s.min()),
            "max": int(s.max()),
            "zero_weeks": int((s == 0).sum()),
        }
        for lag in range(1, MAX_LAG + 1):
            row[f"acf_lag{lag}"] = s.autocorr(lag)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_series(weekly: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, SERIES_NAMES):
        s = weekly[name]
        ax.plot(s.index, s.values, marker="o", ms=3, lw=1, label="unidades semanales")
        ax.plot(s.index, s.rolling(4).mean(), lw=2, label="media móvil 4 semanas")
        ax.axhline(s.mean(), color="gray", ls="--", lw=1, label="media anual")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("unidades / semana")
    fig.suptitle("Ventas semanales por serie (3 categorías originales × género = 6 series), 2023")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_dispersion(weekly: pd.DataFrame, stats: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].boxplot([weekly[n] for n in SERIES_NAMES], tick_labels=SERIES_NAMES)
    axes[0].set_title("Distribución de unidades semanales")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(stats["series"], stats["zero_weeks"])
    axes[1].set_title("Semanas con 0 ventas (de 52)")
    axes[1].tick_params(axis="x", rotation=45)

    # Autocorrelación: con 52 puntos, |r| < 1.96/sqrt(52) ≈ 0.27 no es distinguible de cero.
    band = 1.96 / np.sqrt(len(weekly))
    lags = np.arange(1, MAX_LAG + 1)
    width = 0.13
    for i, name in enumerate(SERIES_NAMES):
        row = stats.loc[stats["series"] == name].iloc[0]
        axes[2].bar(lags + (i - 2.5) * width, [row[f"acf_lag{k}"] for k in lags], width, label=name)
    axes[2].axhspan(-band, band, color="gray", alpha=0.2, label="±1.96/√52 (ruido)")
    axes[2].set_xticks(lags)
    axes[2].set_xlabel("rezago (semanas)")
    axes[2].set_title("Autocorrelación por rezago")
    axes[2].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weekly = weekly_series(load_sales(), target="units")
    stats = series_stats(weekly)
    stats.round(3).to_csv(OUT_DIR / "eda_stats.csv", index=False)
    plot_series(weekly, OUT_DIR / "eda_series.png")
    plot_dispersion(weekly, stats, OUT_DIR / "eda_dispersion.png")

    band = 1.96 / np.sqrt(len(weekly))
    acf = stats[[f"acf_lag{k}" for k in range(1, MAX_LAG + 1)]].to_numpy()
    print(stats.round(2).to_string(index=False))
    print(f"\nBanda de ruido ±{band:.3f}; autocorrelaciones fuera de la banda: {(abs(acf) > band).sum()} de {acf.size}")
    print(f"Archivos en {OUT_DIR}")


if __name__ == "__main__":
    main()
