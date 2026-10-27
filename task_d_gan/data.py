"""Tarea D (Elizabeth): Fashion-MNIST normalizado a [-1, 1] y utilidades para guardar grillas de imágenes.

Por qué [-1, 1]: el generador termina en Tanh, cuya salida está en [-1, 1]. Si las imágenes reales
estuvieran en [0, 1], el discriminador podría distinguirlas solo por la escala (una pista artificial).

Uso: python -m task_d_gan.data   -> outputs/task_d/samples/real_grid.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from task_d_gan.paths import DATA_DIR, SAMPLES_DIR

# Las 10 clases de Fashion-MNIST en el orden de sus etiquetas 0..9 (torchvision: T-shirt/top, Trouser, ...).
CLASSES = ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
           "Sandalia", "Camisa", "Tenis", "Bolso", "Botín"]
CLASSES_EN = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
IMAGE_SIZE = 28


def to_unit_range(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] (N, 28, 28) -> float32 [-1, 1] (N, 1, 28, 28). Misma fórmula en entrenamiento y evaluación."""
    return images_uint8.float().div(127.5).sub(1.0).unsqueeze(1)


def to_uint8(images: torch.Tensor) -> np.ndarray:
    """Inverso para mostrar: [-1, 1] (N, 1, 28, 28) -> uint8 (N, 28, 28). La pantalla usa esta misma función."""
    x = images.detach().clamp(-1, 1).add(1).mul(127.5).round()
    return x.squeeze(1).to(torch.uint8).cpu().numpy()


def load_fashion_mnist(train: bool = True, download: bool = True):
    """Devuelve (X, y): X float32 (N, 1, 28, 28) en [-1, 1] e y int64 (N,) con las etiquetas 0..9.

    Se carga todo en memoria (60 000 × 784 floats ≈ 188 MB): es más rápido que aplicar transforms imagen a imagen.
    """
    from torchvision import datasets  # import diferido: la pantalla no lo necesita

    ds = datasets.FashionMNIST(str(DATA_DIR), train=train, download=download)
    assert list(ds.classes) == CLASSES_EN, ds.classes  # el orden de CLASSES depende de esto
    return to_unit_range(ds.data), ds.targets.clone().long()


def save_grid(images: torch.Tensor, path, nrow: int, title: str | None = None, row_labels=None,
              col_labels=None, cell: float = 0.62, dpi: int = 110) -> None:
    """Grilla de imágenes en escala de grises (fondo negro, como Fashion-MNIST) con títulos opcionales.

    Se escala con 'nearest' (sin interpolar): se ven los 28×28 píxeles reales, sin inventar detalle.
    """
    imgs = to_uint8(images)
    n = len(imgs)
    ncol = nrow
    nrows = int(np.ceil(n / ncol))
    left = 1.25 if row_labels else 0.1
    top = 0.45 if title else 0.1
    top += 0.3 if col_labels else 0
    fig_w, fig_h = left + ncol * cell + 0.1, top + nrows * cell + 0.1
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    for i in range(n):
        r, c = divmod(i, ncol)
        ax = fig.add_axes([(left + c * cell) / fig_w, 1 - (top + (r + 1) * cell) / fig_h,
                           cell * 0.96 / fig_w, cell * 0.96 / fig_h])
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row_labels and c == 0:
            ax.set_ylabel(row_labels[r], rotation=0, ha="right", va="center", fontsize=9, labelpad=6)
        if col_labels and r == 0:
            ax.set_title(col_labels[c], fontsize=8, pad=3)
    if title:
        fig.text(0.5, 1 - 0.25 / fig_h, title, ha="center", va="center", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def examples_per_class(X: torch.Tensor, y: torch.Tensor, per_class: int, seed: int = 0) -> torch.Tensor:
    """`per_class` imágenes de cada clase, ordenadas por clase (filas = clases en save_grid)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for c in range(len(CLASSES)):
        idx = torch.nonzero(y == c).squeeze(1)
        out.append(X[idx[torch.randperm(len(idx), generator=g)[:per_class]]])
    return torch.cat(out)


def main() -> None:
    X, y = load_fashion_mnist(train=True)
    Xt, yt = load_fashion_mnist(train=False)
    print(f"train: {tuple(X.shape)} rango [{X.min():.1f}, {X.max():.1f}] · test: {tuple(Xt.shape)}")
    print("imágenes por clase (train):", torch.bincount(y).tolist())
    path = SAMPLES_DIR / "real_grid.png"
    save_grid(examples_per_class(X, y, 10), path, nrow=10, row_labels=CLASSES,
              title="Fashion-MNIST real (entrenamiento): 10 por clase")
    print("guardado:", path)


if __name__ == "__main__":
    main()
