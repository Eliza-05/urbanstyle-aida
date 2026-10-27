"""Tarea D (Elizabeth): CNN pequeña (semana 11) que clasifica Fashion-MNIST. Solo sirve para EVALUAR las GAN:

- ¿qué prenda generó la DCGAN? -> cobertura de clases (detecta colapso de modos);
- ¿la cGAN generó la prenda pedida? -> matriz pedida vs. predicha;
- sus activaciones de la penúltima capa (128 valores) -> FID aproximado (FID-CNN, ver metrics.py).

Recibe imágenes en [-1, 1], la MISMA escala que producen los generadores (no hay que reescalar nada).
Uso: python -m task_d_gan.classifier   -> models/fashion_cnn.pt
"""
import time

import torch
from torch import nn

from common import config
from task_d_gan.data import CLASSES, load_fashion_mnist
from task_d_gan.paths import CLASSIFIER_CKPT

EPOCHS = 6
BATCH = 128
LR = 1e-3


class FashionCNN(nn.Module):
    """Conv(32) -> MaxPool -> Conv(64) -> MaxPool -> Linear(128) -> Linear(10)."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14 -> 7
            nn.Flatten(),
        )
        self.embed = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU())  # rasgos para el FID-CNN
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@torch.no_grad()
def predict(model: FashionCNN, X: torch.Tensor, batch: int = 1000):
    """(clase predicha, rasgos de 128) para un tensor de imágenes en [-1, 1]."""
    model.eval()
    preds, feats = [], []
    for i in range(0, len(X), batch):
        f = model.features(X[i:i + batch])
        feats.append(f)
        preds.append(model.head(f).argmax(1))
    return torch.cat(preds), torch.cat(feats)


def load_classifier(path=CLASSIFIER_CKPT) -> FashionCNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FashionCNN(len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    config.set_seed()
    X, y = load_fashion_mnist(train=True)
    Xt, yt = load_fashion_mnist(train=False)
    model = FashionCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(config.SEED)
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for idx in torch.randperm(len(X), generator=g).split(BATCH):
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
        acc = (predict(model, Xt)[0] == yt).float().mean().item()
        print(f"época {epoch}: pérdida {loss.item():.3f} · exactitud test {acc:.4f}")
    seconds = time.perf_counter() - start
    pred = predict(model, Xt)[0]
    per_class = [(pred[yt == c] == c).float().mean().item() for c in range(len(CLASSES))]
    CLASSIFIER_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": CLASSES, "test_accuracy": acc,
                "test_accuracy_per_class": per_class, "epochs": EPOCHS, "seed": config.SEED,
                "train_seconds": seconds}, CLASSIFIER_CKPT)
    print(f"exactitud en test {acc:.4f} ({seconds:.0f} s en CPU) -> {CLASSIFIER_CKPT}")
    print("por clase:", {c: round(a, 3) for c, a in zip(CLASSES, per_class)})


if __name__ == "__main__":
    main()
