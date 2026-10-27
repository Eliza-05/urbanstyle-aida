"""Tarea D (Elizabeth): rutas de datos, modelos y salidas (ver outputs/task_d/README.md)."""
from common import config

DATA_DIR = config.ROOT / "data"                    # torchvision descarga aquí data/FashionMNIST/ (gitignored)

OUT_DIR = config.OUTPUTS_DIR / "task_d"
SAMPLES_DIR = OUT_DIR / "samples"                  # grillas de reales y de generadas, interpolaciones
TRAINING_DIR = OUT_DIR / "training"                # curvas, historial por época, grillas por época y GIF
EVALUATION_DIR = OUT_DIR / "evaluation"            # cobertura de clases, control de la cGAN, memorización, FID-CNN
POSTER_DIR = OUT_DIR / "poster"                    # pósteres promocionales (bonus)
FIGURES_DIR = OUT_DIR / "figures"                  # capturas de la pantalla Pygame

GAN_CKPT = config.MODELS_DIR / "gan_generator.pt"      # DCGAN (obligatorio)
CGAN_CKPT = config.MODELS_DIR / "cgan_generator.pt"    # GAN condicional (bonus)
CLASSIFIER_CKPT = config.MODELS_DIR / "fashion_cnn.pt"  # CNN auxiliar SOLO para evaluar (no genera)
EVAL_JSON = EVALUATION_DIR / "summary.json"            # números de la evaluación que lee la pantalla

# Pesos de cada época del entrenamiento exploratorio (para elegir el número de épocas): no se suben a Git.
SNAPSHOT_DIR = config.ROOT / "checkpoints" / "task_d"
