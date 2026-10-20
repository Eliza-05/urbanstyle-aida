"""Tarea B (Elizabeth): ¿conviene reentrenar con train+val antes del pronóstico final? Backtest SIN tocar prueba.

Se repite la situación 8 semanas antes, usando solo las semanas 0-43:
  A) como el modelo actual: entrena con 0-27 y hace early stopping con 28-35;
  B) reentrenado: entrena con 0-35 (escalador ajustado con 0-35) durante las épocas que eligió A.
Ambos predicen las semanas 36-43, que ninguno vio. Se repite con 5 semillas.
Uso: python -m task_b_forecasting.retrain_check  →  outputs/task_b/training/retrain_check.csv
"""
import pandas as pd
import torch

from common import config
from task_b_forecasting.baselines import mae, rmse
from task_b_forecasting.prepare import TEST_WEEKS, VAL_WEEKS, make_windows
from task_b_forecasting.paths import TRAINING_DIR
from task_b_forecasting.train import HPARAMS, load_data, predict_units, train_model, train_per_series

SEEDS = [config.SEED + i for i in range(5)]


def retrain_without_validation(data, histories, seed):
    """B: una LSTM por serie entrenada con todo el tramo de entrenamiento, con las épocas que eligió A."""
    models = []
    for s, hist in enumerate(histories):
        X, y, _, _ = make_windows(data.scaled, HPARAMS["window"], data.split.targets("train"),
                                  series=[s], one_hot=False)
        hp = {**HPARAMS, "max_epochs": hist["best_epoch"] + 1}  # las épocas se cuentan desde 0
        model, _ = train_model(X, y, None, None, seed=seed, **hp)
        models.append(model)
    return models


def main() -> None:
    torch.set_num_threads(1)
    hidden_tail = VAL_WEEKS + TEST_WEEKS  # semanas 36-51 quedan fuera del entrenamiento en ambos casos
    data_a = load_data(val_weeks=VAL_WEEKS, test_weeks=hidden_tail)  # train 0-27, val 28-35
    data_b = load_data(val_weeks=0, test_weeks=hidden_tail)          # train 0-35
    targets = range(data_b.split.train_end, data_b.split.train_end + VAL_WEEKS)  # semanas 36-43

    rows = []
    for seed in SEEDS:
        models_a, histories = train_per_series(data_a, HPARAMS, seed=seed)
        pred_a, s_idx, t_idx = predict_units(models_a, data_a.scaled, HPARAMS["window"], targets, data_a.scaler)
        models_b = retrain_without_validation(data_b, histories, seed)
        pred_b, _, _ = predict_units(models_b, data_b.scaled, HPARAMS["window"], targets, data_b.scaler)
        actual = data_a.values[t_idx, s_idx]
        rows.append({"seed": seed,
                     "A_train_mae": mae(actual, pred_a), "A_train_rmse": rmse(actual, pred_a),
                     "B_trainval_mae": mae(actual, pred_b), "B_trainval_rmse": rmse(actual, pred_b)})

    results = pd.DataFrame(rows)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(TRAINING_DIR / "retrain_check.csv", index=False)
    print(results.round(3).to_string(index=False))
    print(results.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())
    diff = results["B_trainval_mae"] - results["A_train_mae"]
    print(f"B - A en MAE: media {diff.mean():.3f}; semillas en que B mejora: {(diff < 0).sum()} de {len(diff)}")


if __name__ == "__main__":
    main()
