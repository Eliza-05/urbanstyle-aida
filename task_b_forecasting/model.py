"""Tarea B (Elizabeth): modelo LSTM pequeño de pronóstico (LSTM + capa lineal) y su checkpoint."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import config

CHECKPOINT_PATH = config.MODELS_DIR / "lstm_forecast.pt"


class LSTMForecaster(nn.Module):
    """Lee una ventana de semanas de UNA serie y predice (escalada) la semana siguiente.

    Entrada: (batch, window, input_size), con input_size = 1 (las ventas escaladas de la serie).
    input_size se deja como parámetro porque tune.py también prueba un modelo compartido con
    one-hot de la serie (input_size = 1 + n_series). Salida: (batch,).
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 8, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Con una sola capa LSTM, el dropout interno de nn.LSTM no actúa; se aplica antes de la capa lineal.
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # out: (batch, window, hidden)
        last = out[:, -1, :]           # estado oculto tras la última semana de la ventana
        return self.head(self.dropout(last)).squeeze(-1)


def predict_scaled(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Predicciones en escala [0, 1] (modo evaluación: sin dropout ni gradientes)."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def save_checkpoint(models: list[LSTMForecaster], meta: dict, path: Path = CHECKPOINT_PATH) -> None:
    """Un solo archivo con los pesos de las 6 LSTM (una por serie, en el orden de meta["series_names"])
    y todo lo necesario para reutilizarlas igual que en el entrenamiento (escalador, ventana, etc.)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": [m.state_dict() for m in models], **meta}, path)


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> tuple[list[LSTMForecaster], dict]:
    """Reconstruye las LSTM del checkpoint y las deja en modo evaluación."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    models = []
    for state in ckpt["state_dicts"]:
        model = LSTMForecaster(ckpt["input_size"], ckpt["hidden_size"], ckpt["dropout"])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, ckpt
