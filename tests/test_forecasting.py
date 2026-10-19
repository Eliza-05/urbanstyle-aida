"""Pruebas rápidas de la Tarea B (no entrenan ni necesitan el CSV)."""
import numpy as np
import pytest

from task_b_forecasting.baselines import (
    choose_k,
    mae,
    moving_average_forecast,
    naive_forecast,
    rmse,
)
from task_b_forecasting.prepare import (
    chronological_split,
    fit_scaler,
    make_windows,
    scaler_from_minmax,
)

N_WEEKS, N_SERIES, WINDOW = 52, 6, 4


def toy_values():
    """Serie sintética: el valor codifica (semana, serie), así se puede rastrear de dónde viene cada dato."""
    weeks = np.arange(N_WEEKS).reshape(-1, 1)
    return (weeks * 10 + np.arange(N_SERIES)).astype(float)


# ---------- split y escalado ----------

def test_split_is_chronological_and_complete():
    split = chronological_split(N_WEEKS)
    assert (split.train_end, split.val_end) == (36, 44)
    parts = [list(split.targets(p)) for p in ("train", "val", "test")]
    assert parts[0] + parts[1] + parts[2] == list(range(N_WEEKS))  # sin huecos ni solapes, en orden
    assert max(parts[0]) < min(parts[1]) and max(parts[1]) < min(parts[2])


def test_scaler_uses_only_train_weeks():
    values = toy_values()
    split = chronological_split(N_WEEKS)
    scaler = fit_scaler(values, split)
    # El máximo visto por el escalador es el de la última semana de train (35), no el de val/test.
    np.testing.assert_allclose(scaler.data_max_, values[split.train_end - 1])
    np.testing.assert_allclose(scaler.data_min_, values[0])
    # Si se alteran val y test, el escalador no cambia.
    altered = values.copy()
    altered[split.train_end:] = 1e6
    np.testing.assert_allclose(fit_scaler(altered, split).data_max_, scaler.data_max_)


def test_scaler_roundtrip_from_saved_minmax():
    values = toy_values()
    scaler = fit_scaler(values, chronological_split(N_WEEKS))
    rebuilt = scaler_from_minmax(scaler.data_min_.tolist(), scaler.data_max_.tolist())
    np.testing.assert_allclose(rebuilt.transform(values), scaler.transform(values))
    np.testing.assert_allclose(rebuilt.inverse_transform(scaler.transform(values)), values)


# ---------- ventanas ----------

def test_window_shapes_shared_model():
    split = chronological_split(N_WEEKS)
    X, y, s, t = make_windows(toy_values(), WINDOW, split.targets("train"))
    n = (split.train_end - WINDOW) * N_SERIES
    assert X.shape == (n, WINDOW, 1 + N_SERIES)
    assert y.shape == s.shape == t.shape == (n,)
    assert X.dtype == np.float32


def test_window_shapes_single_series():
    X, y, _, _ = make_windows(toy_values(), WINDOW, range(N_WEEKS), series=[2], one_hot=False)
    assert X.shape == (N_WEEKS - WINDOW, WINDOW, 1)


def test_window_content_is_the_previous_weeks():
    values = toy_values()
    X, y, s, t = make_windows(values, WINDOW, [10], series=[3])
    np.testing.assert_allclose(X[0, :, 0], values[6:10, 3])  # semanas 6..9
    assert y[0] == values[10, 3]
    np.testing.assert_allclose(X[0, :, 1:], np.tile(np.eye(N_SERIES)[3], (WINDOW, 1)))  # one-hot de la serie 3


def test_train_windows_never_touch_val_or_test():
    values = toy_values()
    split = chronological_split(N_WEEKS)
    X, y, s, t = make_windows(values, WINDOW, split.targets("train"))
    week_of = lambda v: v // 10  # el valor codifica la semana
    assert week_of(X[:, :, 0]).max() < split.train_end
    assert week_of(y).max() < split.train_end
    assert t.max() == split.train_end - 1


def test_test_windows_target_only_test_weeks():
    split = chronological_split(N_WEEKS)
    _, _, _, t = make_windows(toy_values(), WINDOW, split.targets("test"))
    assert set(t) == set(range(split.val_end, N_WEEKS))


def test_split_too_short_raises():
    with pytest.raises(ValueError):
        chronological_split(10)


# ---------- métricas y líneas base ----------


def test_metrics_by_hand():
    # errores: 1, -2, 3 -> MAE = 6/3 = 2 ; RMSE = sqrt((1+4+9)/3) = sqrt(14/3)
    y_true = [10, 5, 8]
    y_pred = [9, 7, 5]
    assert mae(y_true, y_pred) == pytest.approx(2.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(14 / 3))
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)  # RMSE nunca es menor que MAE


def test_metrics_match_sklearn():
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 20, 50), rng.integers(0, 20, 50)
    assert mae(a, b) == pytest.approx(mean_absolute_error(a, b))
    assert rmse(a, b) == pytest.approx(np.sqrt(mean_squared_error(a, b)))


def test_naive_and_moving_average_by_hand():
    values = np.array([[2.0], [4.0], [6.0], [10.0], [0.0]])
    assert naive_forecast(values, [4])[0, 0] == 10.0
    assert moving_average_forecast(values, [4], k=3)[0, 0] == pytest.approx((4 + 6 + 10) / 3)
    assert moving_average_forecast(values, [3], k=3)[0, 0] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        moving_average_forecast(values, [1], k=3)


def test_choose_k_uses_only_given_targets():
    # Serie constante en 5 hasta la semana 19 y luego 50: con objetivos 10..19 todo k es perfecto,
    # y el salto posterior (como si fuera prueba) no puede influir en la elección.
    values = np.full((30, 1), 5.0)
    values[20:] = 50.0
    best, scores = choose_k(values, range(10, 20), candidates=(2, 4, 8))
    assert all(s == (0.0, 0.0) for s in scores.values())
    assert best == 2
