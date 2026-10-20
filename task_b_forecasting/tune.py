"""Tarea B (Elizabeth): elige con VALIDACIÓN (nunca con prueba) el tipo de modelo y sus hiperparámetros.

Compara (A) una LSTM por serie contra (B) una LSTM compartida con one-hot de la serie, en una grilla
pequeña de ventana, tamaño oculto, dropout y tasa de aprendizaje, con 3 semillas por configuración.
Uso: python -m task_b_forecasting.tune  →  outputs/task_b/training/tuning_results.csv
"""
import itertools
import time

import numpy as np
import pandas as pd
import torch

from task_b_forecasting.baselines import mae, rmse
from task_b_forecasting.model import predict_scaled
from task_b_forecasting.paths import TRAINING_DIR
from task_b_forecasting.train import HPARAMS, load_data, train_model, unscale, windows_for

SEEDS = (0, 1, 2)
GRID = {"window": (4, 8), "hidden_size": (2, 4, 8, 16, 32), "dropout": (0.0, 0.2), "lr": (0.001, 0.005)}


def val_predictions(data, hp, seed, shared: bool):
    """Predicciones de validación en unidades reales para el modelo compartido (B) o por serie (A)."""
    if shared:
        X_tr, y_tr, _, _ = windows_for(data, hp["window"], "train")
        X_va, y_va, s_va, t_va = windows_for(data, hp["window"], "val")
        model, hist = train_model(X_tr, y_tr, X_va, y_va, seed=seed, **hp)
        pred = unscale(predict_scaled(model, X_va), s_va, data.scaler)
        return pred, data.values[t_va, s_va], [hist["best_epoch"]]
    preds, actual, epochs = [], [], []
    for s in range(data.values.shape[1]):
        X_tr, y_tr, _, _ = windows_for(data, hp["window"], "train", series=[s], one_hot=False)
        X_va, y_va, s_va, t_va = windows_for(data, hp["window"], "val", series=[s], one_hot=False)
        model, hist = train_model(X_tr, y_tr, X_va, y_va, seed=seed, **hp)
        preds.append(unscale(predict_scaled(model, X_va), s_va, data.scaler))
        actual.append(data.values[t_va, s_va])
        epochs.append(hist["best_epoch"])
    return np.concatenate(preds), np.concatenate(actual), epochs


def main() -> None:
    torch.set_num_threads(1)  # modelos diminutos: un hilo es más rápido y estable
    data = load_data()
    rows = []
    t0 = time.time()
    for shared, (window, hidden, dropout, lr) in itertools.product((False, True), itertools.product(*GRID.values())):
        hp = {**HPARAMS, "window": window, "hidden_size": hidden, "dropout": dropout, "lr": lr}
        maes, rmses, epochs = [], [], []
        for seed in SEEDS:
            pred, actual, ep = val_predictions(data, hp, seed, shared)
            maes.append(mae(actual, pred))
            rmses.append(rmse(actual, pred))
            epochs += ep
        rows.append({
            "model": "B_compartido" if shared else "A_por_serie",
            "window": window, "hidden_size": hidden, "dropout": dropout, "lr": lr,
            "val_mae_mean": np.mean(maes), "val_mae_std": np.std(maes),
            "val_rmse_mean": np.mean(rmses), "best_epoch_mean": np.mean(epochs),
        })
        print(rows[-1])
    results = pd.DataFrame(rows).sort_values("val_mae_mean")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(TRAINING_DIR / "tuning_results.csv", index=False)
    print(f"\n{time.time() - t0:.0f} s\n", results.round(3).to_string(index=False))
    print("\nMejor por tipo de modelo:")
    print(results.groupby("model").head(1).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
