"""Tarea B (Elizabeth): métricas (MAE, RMSE) y líneas base de pronóstico: ingenua y media móvil.

Todo trabaja en unidades reales (sin escalar). `values` es la matriz (semanas × series).
"""
import numpy as np

K_CANDIDATES = (2, 3, 4, 6, 8, 12)


def mae(y_true, y_pred) -> float:
    """Error absoluto medio: cuántas unidades se equivoca en promedio."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Raíz del error cuadrático medio: castiga más los errores grandes."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def naive_forecast(values: np.ndarray, targets) -> np.ndarray:
    """Predice que la semana t vende lo mismo que la semana t-1. Devuelve (len(targets), series)."""
    return np.array([values[t - 1] for t in targets], dtype=float)


def moving_average_forecast(values: np.ndarray, targets, k: int) -> np.ndarray:
    """Predice la semana t como el promedio de las k semanas anteriores (t-k ... t-1)."""
    if min(targets) < k:
        raise ValueError("No hay k semanas de historia para todas las semanas objetivo")
    return np.array([values[t - k : t].mean(axis=0) for t in targets], dtype=float)


def choose_k(values: np.ndarray, val_targets, candidates=K_CANDIDATES) -> tuple[int, dict]:
    """Elige el k de la media móvil con menor MAE en VALIDACIÓN (nunca con prueba).

    Se usa un mismo k para las 6 series: con 8 semanas de validación por serie, elegir un k por
    serie sería sobreajustar la línea base. Devuelve (mejor k, {k: (mae, rmse)}).
    """
    actual = values[list(val_targets)]
    scores = {}
    for k in candidates:
        pred = moving_average_forecast(values, val_targets, k)
        scores[k] = (mae(actual, pred), rmse(actual, pred))
    best = min(scores, key=lambda k: scores[k])  # desempata por RMSE
    return best, scores
