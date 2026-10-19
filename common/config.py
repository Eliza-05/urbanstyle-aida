"""Configuración global: rutas relativas a la raíz del repo, semilla y constantes de UI."""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
SALES_CSV = DATA_RAW / "retail_sales_dataset.csv"

SEED = 42

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 640
FPS = 60

COLOR_BG = (24, 26, 32)
COLOR_TEXT = (235, 235, 240)
COLOR_BUTTON = (60, 66, 84)
COLOR_BUTTON_HOVER = (86, 96, 128)
COLOR_ACCENT = (120, 170, 255)


def set_seed(seed: int = SEED) -> None:
    """Fija las semillas de random, numpy y torch."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
