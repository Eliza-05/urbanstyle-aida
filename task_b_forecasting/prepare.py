"""Tarea B (Elizabeth): split cronológico, escalado MinMax (ajustado solo con train) y ventanas deslizantes.

Convención: `values` es una matriz (semanas × series) en orden cronológico. Un "objetivo" t es la semana
que se quiere predecir; su entrada son las `window` semanas anteriores (t-window ... t-1).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 52 semanas = 36 de entrenamiento + 8 de validación + 8 de prueba (las últimas).
VAL_WEEKS = 8
TEST_WEEKS = 8


@dataclass
class Split:
    """Límites del split cronológico: train = [0, train_end), val = [train_end, val_end), test = [val_end, n)."""
    train_end: int
    val_end: int
    n_weeks: int

    def targets(self, part: str) -> range:
        """Semanas objetivo de cada parte."""
        return {
            "train": range(0, self.train_end),
            "val": range(self.train_end, self.val_end),
            "test": range(self.val_end, self.n_weeks),
        }[part]


def chronological_split(n_weeks: int, val_weeks: int = VAL_WEEKS, test_weeks: int = TEST_WEEKS) -> Split:
    """Las últimas semanas son prueba, las anteriores validación y el resto entrenamiento (nunca se baraja)."""
    train_end = n_weeks - val_weeks - test_weeks
    if train_end <= 0:
        raise ValueError("No quedan semanas para entrenar")
    return Split(train_end=train_end, val_end=n_weeks - test_weeks, n_weeks=n_weeks)


def fit_scaler(values: np.ndarray, split: Split) -> MinMaxScaler:
    """MinMax por serie (una columna = una serie), ajustado SOLO con las semanas de entrenamiento."""
    return MinMaxScaler().fit(values[: split.train_end])


def scaler_from_minmax(data_min, data_max) -> MinMaxScaler:
    """Reconstruye un MinMaxScaler idéntico a partir del mínimo y máximo guardados en el checkpoint."""
    return MinMaxScaler().fit(np.array([data_min, data_max], dtype=float))


def make_windows(scaled: np.ndarray, window: int, targets, series=None, one_hot: bool = True):
    """Convierte la serie en pares (entrada, objetivo) con una ventana deslizante.

    - scaled: matriz (semanas × series) ya escalada.
    - targets: semanas objetivo t; solo se usan las que tienen `window` semanas previas (t >= window).
    - series: índices de las series a incluir (por defecto todas).
    - one_hot: si es True, cada paso de la entrada lleva además el one-hot de la serie (modelo compartido).

    Devuelve X (n, window, n_features), y (n,), series_idx (n,), target_week (n,).
    """
    n_series = scaled.shape[1]
    series = range(n_series) if series is None else series
    X, y, s_idx, t_idx = [], [], [], []
    for s in series:
        for t in targets:
            if t < window:
                continue
            past = scaled[t - window : t, s].reshape(window, 1)
            if one_hot:
                code = np.zeros((window, n_series))
                code[:, s] = 1.0
                past = np.hstack([past, code])
            X.append(past)
            y.append(scaled[t, s])
            s_idx.append(s)
            t_idx.append(t)
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(s_idx, dtype=np.int64),
        np.array(t_idx, dtype=np.int64),
    )
