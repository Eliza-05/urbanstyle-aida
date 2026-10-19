"""Pruebas rápidas de la Tarea B (no entrenan ni necesitan el CSV)."""
import numpy as np
import pandas as pd
import pytest
import torch

from common import config
from task_b_forecasting.baselines import (
    choose_k,
    mae,
    moving_average_forecast,
    naive_forecast,
    rmse,
)
from task_b_forecasting.evaluate import build_predictions, forecast_next_week, metrics_table
from task_b_forecasting.model import LSTMForecaster, load_checkpoint, predict_scaled, save_checkpoint
from task_b_forecasting.prepare import (
    TEST_WEEKS,
    chronological_split,
    fit_scaler,
    make_windows,
    scaler_from_minmax,
)
from task_b_forecasting.train import Data, predict_units

N_WEEKS, N_SERIES, WINDOW = 52, 6, 4


def toy_values():
    """Serie sintética: el valor codifica (semana, serie), así se puede rastrear de dónde viene cada dato."""
    weeks = np.arange(N_WEEKS).reshape(-1, 1)
    return (weeks * 10 + np.arange(N_SERIES)).astype(float)


# ---------- split y escalado ----------

def test_split_is_chronological_and_complete():
    split = chronological_split(N_WEEKS)
    assert split.train_end == 36
    assert split.val_end == 44
    parts = [list(split.targets(p)) for p in ("train", "val", "test")]
    assert parts[0] + parts[1] + parts[2] == list(range(N_WEEKS))  # sin huecos ni solapes, en orden
    assert max(parts[0]) < min(parts[1])
    assert max(parts[1]) < min(parts[2])


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


# ---------- modelo, checkpoint y evaluación (sin entrenar) ----------


def untrained_models(n=N_SERIES, hidden=4):
    torch.manual_seed(0)
    return [LSTMForecaster(1, hidden, 0.2) for _ in range(n)]


def fake_data():
    values = toy_values() % 23  # valores tipo "unidades" entre 0 y 22
    weekly = pd.DataFrame(values, index=pd.date_range("2023-01-01", periods=N_WEEKS, freq="7D"),
                          columns=[f"S{i}" for i in range(N_SERIES)])
    split = chronological_split(N_WEEKS)
    scaler = fit_scaler(values, split)
    return Data(weekly, values, split, scaler, scaler.transform(values))


def fake_ckpt(data, window=WINDOW, k=3):
    return {"input_size": 1, "hidden_size": 4, "dropout": 0.2, "window": window, "ma_k": k,
            "series_names": list(data.weekly.columns), "scaler_min": data.scaler.data_min_.tolist(),
            "scaler_max": data.scaler.data_max_.tolist(), "train_end": data.split.train_end,
            "val_end": data.split.val_end, "n_weeks": data.split.n_weeks}


def test_model_output_shape():
    model = LSTMForecaster(1, 8, 0.2)
    out = predict_scaled(model, np.zeros((5, WINDOW, 1), dtype=np.float32))
    assert out.shape == (5,)


def test_checkpoint_roundtrip_reproduces_predictions(tmp_path):
    data = fake_data()
    models = untrained_models()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(models, fake_ckpt(data), path)
    reloaded, meta = load_checkpoint(path)
    assert meta["window"] == WINDOW
    assert len(reloaded) == N_SERIES
    targets = data.split.targets("test")
    before = predict_units(models, data.scaled, WINDOW, targets, data.scaler)
    after = predict_units(reloaded, data.scaled, WINDOW, targets, data.scaler)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_predict_units_inverts_scaling_per_series():
    data = fake_data()
    models = untrained_models()
    pred, s_idx, t_idx = predict_units(models, data.scaled, WINDOW, [40, 41], data.scaler)
    assert pred.shape == (2 * N_SERIES,)
    assert list(s_idx[:2]) == [0, 0]
    assert list(t_idx[:2]) == [40, 41]
    # Cada predicción en unidades = salida cruda de SU red × rango + mínimo de SU serie.
    for s, model in enumerate(models):
        X, _, _, _ = make_windows(data.scaled, WINDOW, [40, 41], series=[s], one_hot=False)
        raw = predict_scaled(model, X)
        expected = raw * data.scaler.data_range_[s] + data.scaler.data_min_[s]
        np.testing.assert_allclose(pred[s_idx == s], expected, rtol=1e-6)


def test_build_predictions_aligns_models_on_same_weeks():
    data = fake_data()
    ckpt = fake_ckpt(data, window=WINDOW, k=3)
    pred = build_predictions(untrained_models(), ckpt, data)
    test = pred[pred["part"] == "test"]
    assert len(test) == TEST_WEEKS * N_SERIES
    row = test[(test["series"] == "S2")].iloc[0]  # primera semana de prueba (t = 44) de la serie 2
    np.testing.assert_allclose(row["actual"], data.values[44, 2])
    np.testing.assert_allclose(row["moving_average"], data.values[41:44, 2].mean())
    np.testing.assert_allclose(row["naive"], data.values[43, 2])
    metrics = metrics_table(pred)
    assert list(metrics["series"])[-1] == "GLOBAL"
    assert metrics.set_index("series").loc["GLOBAL", "n"] == TEST_WEEKS * N_SERIES


def test_forecast_next_week_is_the_week_after_the_data():
    data = fake_data()
    nxt = forecast_next_week(untrained_models(), fake_ckpt(data, k=3), data)
    assert len(nxt) == N_SERIES
    assert nxt["week_start"].iloc[0] == (data.weekly.index[-1] + pd.Timedelta(days=7)).date()
    np.testing.assert_allclose(nxt["moving_average"], data.values[-3:].mean(axis=0))


@pytest.mark.skipif(not (config.SALES_CSV.exists() and (config.MODELS_DIR / "lstm_forecast.pt").exists()),
                    reason="requiere el CSV y el checkpoint entrenado")
def test_saved_checkpoint_and_screen_headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame
    from task_b_forecasting import screen as screen_b
    from task_b_forecasting.evaluate import load_results
    results = load_results()
    ckpt = results["ckpt"]
    assert ckpt["series_names"] == ["Clothing-Male", "Clothing-Female", "Beauty-Male",
                                    "Beauty-Female", "Electronics-Male", "Electronics-Female"]
    assert len(ckpt["state_dicts"]) == 6
    assert results["metrics"].shape[0] == 7  # 6 series + global
    pygame.init()
    surface = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    view = screen_b.ForecastScreen(results)
    view.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    assert view.series_name == "Clothing-Female"
    view.draw(surface)
    view.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert view.done
    pygame.quit()
