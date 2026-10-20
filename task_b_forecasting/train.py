"""Tarea B (Elizabeth): entrena una LSTM pequeña por serie (6 en total) y las guarda en un checkpoint único.

Uso: python -m task_b_forecasting.train
Salidas: models/lstm_forecast.pt, outputs/task_b/training/loss_curve.png y loss_history.csv
"""
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from common import config
from common.data import SERIES_NAMES, TARGETS, WEEK_START, load_sales, weekly_series
from task_b_forecasting.baselines import choose_k
from task_b_forecasting.model import (
    CHECKPOINT_PATH,
    LSTMForecaster,
    load_checkpoint,
    predict_scaled,
    save_checkpoint,
)
from task_b_forecasting.paths import TRAINING_DIR
from task_b_forecasting.prepare import (
    TEST_WEEKS,
    VAL_WEEKS,
    Split,
    chronological_split,
    fit_scaler,
    make_windows,
)

TARGET = "units"

# Configuración elegida con VALIDACIÓN en task_b_forecasting/tune.py (docs/notas_informe_tarea_b.md §6).
HPARAMS = {
    "window": 4,         # semanas de entrada
    "hidden_size": 8,    # unidades ocultas (361 parámetros por LSTM; con 2 la red casi no entrena, ver notas §6.3)
    "dropout": 0.2,
    "lr": 0.005,         # Adam
    "batch_size": 32,
    "max_epochs": 300,
    "patience": 30,      # early stopping
}


@dataclass
class Data:
    """Datos listos para modelar: tabla semanal, valores reales, split, escalador y valores escalados."""
    weekly: pd.DataFrame
    values: np.ndarray
    split: Split
    scaler: object
    scaled: np.ndarray


def load_data(target: str = TARGET, val_weeks: int = VAL_WEEKS, test_weeks: int = TEST_WEEKS) -> Data:
    """Tabla semanal + split cronológico + escalador ajustado solo con las semanas de entrenamiento."""
    weekly = weekly_series(load_sales(), target=target)
    values = weekly.to_numpy(dtype=float)
    split = chronological_split(len(values), val_weeks, test_weeks)
    scaler = fit_scaler(values, split)
    return Data(weekly, values, split, scaler, scaler.transform(values))


def unscale(pred_scaled: np.ndarray, series_idx: np.ndarray, scaler) -> np.ndarray:
    """Pasa predicciones escaladas a unidades reales usando el mínimo y rango de la serie correspondiente."""
    return pred_scaled * scaler.data_range_[series_idx] + scaler.data_min_[series_idx]


def windows_for(data: Data, window: int, part: str, series=None, one_hot: bool = True):
    return make_windows(data.scaled, window, data.split.targets(part), series=series, one_hot=one_hot)


def _train_one_epoch(model, optimizer, loss_fn, X, y, batch_size) -> float:
    """Una pasada por los datos en mini-lotes; devuelve la pérdida media de la época."""
    model.train()
    order = torch.randperm(len(X))  # se barajan los EJEMPLOS (ventanas), no las semanas
    total = 0.0
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    return total / len(order)


def train_model(X_tr, y_tr, X_va, y_va, *, hidden_size, dropout, lr, batch_size, max_epochs, patience,
                seed=config.SEED, **_):
    """Ciclo de entrenamiento con Adam + MSE y early stopping sobre la pérdida de validación.

    Devuelve el modelo con los pesos de la MEJOR época de validación y el historial de pérdidas.
    Si X_va es None (reentrenar sin validación) entrena exactamente `max_epochs` épocas.
    """
    config.set_seed(seed)
    model = LSTMForecaster(X_tr.shape[2], hidden_size, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_tr_t, y_tr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    history = {"train_loss": [], "val_loss": []}

    if X_va is None:
        for _ in range(max_epochs):
            history["train_loss"].append(_train_one_epoch(model, optimizer, loss_fn, X_tr_t, y_tr_t, batch_size))
        model.eval()
        history["best_epoch"] = max_epochs - 1
        return model, history

    X_va_t, y_va_t = torch.from_numpy(X_va), torch.from_numpy(y_va)
    best_val, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        history["train_loss"].append(_train_one_epoch(model, optimizer, loss_fn, X_tr_t, y_tr_t, batch_size))
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break  # early stopping: la validación no mejora hace `patience` épocas

    model.load_state_dict(best_state)
    model.eval()
    history["best_epoch"] = best_epoch
    return model, history


def train_per_series(data: Data, hp: dict, seed: int = config.SEED):
    """Entrena una LSTM independiente para cada una de las 6 series (entrada: solo esa serie)."""
    models, histories = [], []
    for s in range(data.values.shape[1]):
        X_tr, y_tr, _, _ = windows_for(data, hp["window"], "train", series=[s], one_hot=False)
        X_va, y_va, _, _ = windows_for(data, hp["window"], "val", series=[s], one_hot=False)
        model, history = train_model(X_tr, y_tr, X_va, y_va, seed=seed, **hp)
        models.append(model)
        histories.append(history)
    return models, histories


def predict_units(models, scaled: np.ndarray, window: int, targets, scaler):
    """Predicción en unidades reales de cada serie con su propia LSTM.

    Devuelve (pred, series_idx, target_week), con una fila por (serie, semana objetivo).
    """
    preds, s_all, t_all = [], [], []
    for s, model in enumerate(models):
        X, _, s_idx, t_idx = make_windows(scaled, window, targets, series=[s], one_hot=False)
        preds.append(unscale(predict_scaled(model, X), s_idx, scaler))
        s_all.append(s_idx)
        t_all.append(t_idx)
    return np.concatenate(preds), np.concatenate(s_all), np.concatenate(t_all)


def plot_loss(histories, path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for ax, name, h in zip(axes.flat, SERIES_NAMES, histories):
        ax.plot(h["train_loss"], label="entrenamiento")
        ax.plot(h["val_loss"], label="validación")
        ax.axvline(h["best_epoch"], color="gray", ls="--", label=f"mejor época ({h['best_epoch']})")
        ax.set_title(name)
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (escala MinMax)")
    fig.suptitle("Curvas de pérdida de las 6 LSTM (early stopping en la mejor época de validación)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)  # modelos diminutos: un hilo es más rápido
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    t0 = time.time()
    models, histories = train_per_series(data, HPARAMS, seed=config.SEED)
    elapsed = time.time() - t0

    # Todo lo que evaluate.py y la pantalla necesitan para usar los modelos igual que aquí.
    ma_k, _ = choose_k(data.values, data.split.targets("val"))
    meta = {
        "input_size": 1,
        "hidden_size": HPARAMS["hidden_size"],
        "dropout": HPARAMS["dropout"],
        "window": HPARAMS["window"],
        "series_names": SERIES_NAMES,
        "scaler_min": data.scaler.data_min_.tolist(),
        "scaler_max": data.scaler.data_max_.tolist(),
        "target": TARGET,
        "target_column": TARGETS[TARGET],
        "week_start": WEEK_START,
        "first_week": str(data.weekly.index[0].date()),
        "n_weeks": data.split.n_weeks,
        "train_end": data.split.train_end,
        "val_end": data.split.val_end,
        "ma_k": ma_k,
        "hparams": HPARAMS,
        "seed": config.SEED,
        "best_epochs": [h["best_epoch"] for h in histories],
        "epochs_run": [len(h["train_loss"]) for h in histories],
    }
    save_checkpoint(models, meta)

    long = []
    for name, h in zip(SERIES_NAMES, histories):
        long.append(pd.DataFrame({"series": name, "epoch": range(len(h["train_loss"])),
                                  "train_loss": h["train_loss"], "val_loss": h["val_loss"]}))
    pd.concat(long).round(6).to_csv(TRAINING_DIR / "loss_history.csv", index=False)
    plot_loss(histories, TRAINING_DIR / "loss_curve.png")

    # Comprobación: el checkpoint recargado reproduce exactamente las mismas predicciones.
    reloaded, _ = load_checkpoint()
    targets = data.split.targets("val")
    before = predict_units(models, data.scaled, HPARAMS["window"], targets, data.scaler)[0]
    after = predict_units(reloaded, data.scaled, HPARAMS["window"], targets, data.scaler)[0]

    print(f"Entrenamiento de 6 LSTM: {elapsed:.1f} s (CPU)")
    for name, h in zip(SERIES_NAMES, histories):
        print(f"  {name:20s} épocas={len(h['train_loss']):3d} mejor={h['best_epoch']:3d} "
              f"val_loss={min(h['val_loss']):.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(config.ROOT)} — recarga reproduce predicciones: "
          f"{np.array_equal(before, after)}")


if __name__ == "__main__":
    main()
