"""Tarea B (Elizabeth): evalúa el checkpoint contra las líneas base, pronostica la próxima semana y exporta CSV.

Uso: python -m task_b_forecasting.evaluate [--seeds 5]
Salidas en outputs/task_b/: predictions.csv, metrics.csv, next_week.csv, forecast_test.png,
metrics_comparison.png y (con --seeds) seed_results.csv.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import config
from task_b_forecasting.baselines import mae, moving_average_forecast, naive_forecast, rmse
from task_b_forecasting.model import load_checkpoint
from task_b_forecasting.prepare import scaler_from_minmax
from task_b_forecasting.train import HPARAMS, OUT_DIR, load_data, predict_units, train_per_series

MODELS = ("lstm", "moving_average", "naive")
LABELS = {"lstm": "LSTM", "moving_average": "Media móvil", "naive": "Ingenua"}


def part_of(t: int, ckpt: dict) -> str:
    if t < ckpt["train_end"]:
        return "train"
    return "val" if t < ckpt["val_end"] else "test"


def build_predictions(models, ckpt: dict, data) -> pd.DataFrame:
    """Predicciones de LSTM, media móvil e ingenua para cada semana con historia suficiente (formato largo).

    Todas en unidades reales. Las métricas oficiales usan solo las filas con part == "test".
    """
    scaler = scaler_from_minmax(ckpt["scaler_min"], ckpt["scaler_max"])
    scaled = scaler.transform(data.values)
    window, k = ckpt["window"], ckpt["ma_k"]
    first = max(window, k)  # misma semana inicial para los tres modelos
    targets = range(first, len(data.values))

    lstm, s_idx, t_idx = predict_units(models, scaled, window, targets, scaler)
    ma = moving_average_forecast(data.values, targets, k)
    naive = naive_forecast(data.values, targets)
    pos = t_idx - first  # fila de cada objetivo dentro de `targets`

    return pd.DataFrame({
        "week_start": data.weekly.index[t_idx].date,
        "series": [ckpt["series_names"][s] for s in s_idx],
        "part": [part_of(t, ckpt) for t in t_idx],
        "actual": data.values[t_idx, s_idx],
        "lstm": lstm,
        "moving_average": ma[pos, s_idx],
        "naive": naive[pos, s_idx],
    })


def forecast_next_week(models, ckpt: dict, data) -> pd.DataFrame:
    """Pronóstico de la semana siguiente a la última semana completa (sin valor real conocido)."""
    scaler = scaler_from_minmax(ckpt["scaler_min"], ckpt["scaler_max"])
    scaled = scaler.transform(data.values)
    n = len(data.values)
    # Se añade una fila vacía para que la semana n sea un objetivo válido de make_windows.
    padded = np.vstack([scaled, np.zeros((1, scaled.shape[1]))])
    lstm, _, _ = predict_units(models, padded, ckpt["window"], [n], scaler)
    ma = data.values[n - ckpt["ma_k"] :].mean(axis=0)
    week = data.weekly.index[-1] + pd.Timedelta(days=7)
    return pd.DataFrame({
        "week_start": week.date(),
        "series": ckpt["series_names"],
        "lstm": lstm,
        "moving_average": ma,
        "naive": data.values[-1],
    })


def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE y RMSE por serie y global (todas las series juntas), solo en semanas de prueba."""
    test = pred[pred["part"] == "test"]
    rows = []
    for name, group in list(test.groupby("series", sort=False)) + [("GLOBAL", test)]:
        row = {"series": name, "n": len(group)}
        for m in MODELS:
            row[f"{m}_mae"] = mae(group["actual"], group[m])
            row[f"{m}_rmse"] = rmse(group["actual"], group[m])
        rows.append(row)
    return pd.DataFrame(rows)


def verify_with_sklearn(pred: pd.DataFrame, metrics: pd.DataFrame) -> bool:
    """Recalcula las métricas globales con sklearn de forma independiente."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    test = pred[pred["part"] == "test"]
    glob = metrics.set_index("series").loc["GLOBAL"]
    return all(
        np.isclose(glob[f"{m}_mae"], mean_absolute_error(test["actual"], test[m]))
        and np.isclose(glob[f"{m}_rmse"], np.sqrt(mean_squared_error(test["actual"], test[m])))
        for m in MODELS
    )


def load_results():
    """Lo que necesita la pantalla: checkpoint, predicciones, métricas y próxima semana (no entrena)."""
    models, ckpt = load_checkpoint()
    data = load_data(ckpt["target"])
    if not np.allclose(data.scaler.data_min_, ckpt["scaler_min"]) or not np.allclose(
            data.scaler.data_max_, ckpt["scaler_max"]):
        raise RuntimeError("Los datos no coinciden con los del entrenamiento; vuelve a ejecutar train.py")
    pred = build_predictions(models, ckpt, data)
    return {
        "ckpt": ckpt,
        "weekly": data.weekly,
        "predictions": pred,
        "metrics": metrics_table(pred),
        "next_week": forecast_next_week(models, ckpt, data),
    }


def export_predictions(results, path=OUT_DIR / "predictions.csv"):
    """Bonus: exporta las predicciones (semanas evaluadas + próxima semana) a CSV."""
    nxt = results["next_week"].assign(part="next_week", actual=np.nan)
    table = pd.concat([results["predictions"], nxt[results["predictions"].columns]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(path, index=False)
    return path


def seed_study(seeds) -> pd.DataFrame:
    """Reentrena la configuración elegida con varias semillas y mide en prueba (solo para reportar variabilidad)."""
    data = load_data()
    _, ckpt = load_checkpoint()
    rows = []
    for seed in seeds:
        models, hists = train_per_series(data, HPARAMS, seed=seed)
        metrics = metrics_table(build_predictions(models, ckpt, data)).set_index("series").loc["GLOBAL"]
        rows.append({"seed": seed, "best_epoch_mean": np.mean([h["best_epoch"] for h in hists]),
                     **{c: metrics[c] for c in metrics.index if c != "n"}})
    return pd.DataFrame(rows)


def plot_forecasts(results, path) -> None:
    pred, weekly, nxt = results["predictions"], results["weekly"], results["next_week"]
    ckpt = results["ckpt"]
    test_start = weekly.index[ckpt["val_end"]]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, ckpt["series_names"]):
        p = pred[pred["series"] == name]
        weeks = pd.to_datetime(p["week_start"])
        ax.plot(weekly.index, weekly[name], color="black", lw=1, marker="o", ms=2.5, label="real")
        ax.plot(weeks, p["lstm"], color="tab:blue", lw=2, label="LSTM")
        ax.plot(weeks, p["moving_average"], color="tab:orange", lw=1.5, ls="--", label=f"media móvil k={ckpt['ma_k']}")
        n = nxt[nxt["series"] == name].iloc[0]
        ax.plot(pd.Timestamp(n["week_start"]), n["lstm"], "*", color="tab:blue", ms=12, label="próxima semana (LSTM)")
        ax.axvspan(test_start, weekly.index[-1] + pd.Timedelta(days=3), color="gray", alpha=0.15)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("unidades / semana")
    fig.suptitle("Real vs. predicho (zona gris = 8 semanas de prueba). 3 categorías originales × género = 6 series")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(metrics))
    width = 0.27
    for ax, metric in zip(axes, ("mae", "rmse")):
        for i, m in enumerate(MODELS):
            ax.bar(x + (i - 1) * width, metrics[f"{m}_{metric}"], width, label=LABELS[m])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics["series"], rotation=30, ha="right")
        ax.set_title(f"{metric.upper()} en prueba (unidades/semana)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=0, help="reentrenar con N semillas para medir variabilidad")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    metrics = results["metrics"]
    metrics.round(3).to_csv(OUT_DIR / "metrics.csv", index=False)
    results["next_week"].round(2).to_csv(OUT_DIR / "next_week.csv", index=False)
    export_predictions(results)
    plot_forecasts(results, OUT_DIR / "forecast_test.png")
    plot_metrics(metrics, OUT_DIR / "metrics_comparison.png")

    pd.set_option("display.width", 200)
    print("Métricas en prueba (8 semanas × 6 series, unidades reales):")
    print(metrics.round(3).to_string(index=False))
    print("Verificación independiente con sklearn:", verify_with_sklearn(results["predictions"], metrics))
    print("\nPronóstico de la próxima semana:")
    print(results["next_week"].round(2).to_string(index=False))

    if args.seeds:
        seeds = [config.SEED + i for i in range(args.seeds)]
        study = seed_study(seeds)
        study.round(4).to_csv(OUT_DIR / "seed_results.csv", index=False)
        print(f"\nVariabilidad entre semillas {seeds} (métricas globales en prueba):")
        print(study.round(3).to_string(index=False))
        print(study.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
