"""Tarea D (Elizabeth): entrena la DCGAN (obligatorio) y la cGAN (bonus) con el esquema del taller W2_DCGAN.

Cada iteración alterna dos pasos (objetivo minimax  min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]):
  1) Paso de D: reales con etiqueta 1 y falsas con etiqueta 0. Las falsas entran con fake.detach(): se corta
     el grafo hacia G, así este paso solo actualiza D.
  2) Paso de G: pérdida NO saturante, BCE(D(G(z)), 1) = −log D(G(z)). Minimizar log(1 − D(G(z))) daría
     gradientes casi nulos al inicio, cuando D rechaza con facilidad las falsas.
Adam(lr=2e-4, betas=(0.5, 0.999)) para ambos, como en el taller.

Uso:
  python -m task_d_gan.train                     # entrena las dos (las épocas de EPOCHS)  ~6 min en CPU
  python -m task_d_gan.train --model dcgan       # solo una
  python -m task_d_gan.train --explore 30        # exploración para ELEGIR las épocas (no toca models/)
Salidas: models/gan_generator.pt, models/cgan_generator.pt y outputs/task_d/training/
  ({modelo}_history.csv, {modelo}_curves.png, {modelo}_epochs/epoch_XX.png, {modelo}_progress.gif)
Requiere la CNN auxiliar (python -m task_d_gan.classifier) para el monitoreo por época; sin ella entrena igual.
"""
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from common import config
from task_d_gan.data import CLASSES, load_fashion_mnist, save_grid
from task_d_gan.metrics import class_entropy, fid, gaussian_stats
from task_d_gan.models import (
    BASE,
    NZ,
    CondDiscriminator,
    CondGenerator,
    Discriminator,
    Generator,
    count_params,
    generate,
    generate_from_seed,
    load_generator,
    noise,
    save_checkpoint,
)
from task_d_gan.paths import CGAN_CKPT, CLASSIFIER_CKPT, GAN_CKPT, SNAPSHOT_DIR, TRAINING_DIR

HPARAMS = {"nz": NZ, "base": BASE, "batch_size": 128, "lr": 2e-4, "betas": (0.5, 0.999)}
# Épocas elegidas con la exploración (python -m task_d_gan.train --explore 30); ver docs/notas_informe_tarea_d.md.
EPOCHS = {"dcgan": 20, "cgan": 20}
NAMES = {"dcgan": "DCGAN", "cgan": "cGAN"}
CKPTS = {"dcgan": GAN_CKPT, "cgan": CGAN_CKPT}
N_MONITOR = 5000                       # imágenes generadas por época para el FID-CNN y la cobertura de clases
COLORS = {"D": "#2B6CB0", "G": "#D9480F"}  # paleta validada (contraste y daltonismo)


def build(kind: str):
    if kind == "cgan":
        return CondGenerator(len(CLASSES), NZ, BASE), CondDiscriminator(len(CLASSES), BASE)
    return Generator(NZ, BASE), Discriminator(BASE)


def fixed_inputs(kind: str):
    """Ruido fijo (fixed_noise) para comparar las mismas muestras época tras época.

    cGAN: 10 filas (una por clase) × 8 columnas; cada columna repite el MISMO z en todas las clases, así se ve
    que la etiqueta decide la prenda y z decide el "estilo" (grosor, brillo, forma).
    """
    if kind == "dcgan":
        return noise(config.SEED, 64), None
    return noise(config.SEED, 8).repeat(len(CLASSES), 1), torch.arange(len(CLASSES)).repeat_interleave(8)


class Monitor:
    """Calidad por época con la CNN auxiliar: FID-CNN contra las 10 000 reales de prueba, entropía de clases
    (colapso de modos) y, en la cGAN, el % de imágenes clasificadas como la clase pedida."""

    def __init__(self):
        self.clf = None
        if CLASSIFIER_CKPT.exists():
            from task_d_gan.classifier import load_classifier, predict
            self.clf, self.predict = load_classifier(), predict
            Xt, _ = load_fashion_mnist(train=False)
            self.real_stats = gaussian_stats(self.predict(self.clf, Xt)[1])

    def __call__(self, G, conditional: bool) -> dict:
        if self.clf is None:
            return {}
        labels = torch.arange(N_MONITOR) % len(CLASSES) if conditional else None  # 500 por clase
        images = generate(G, noise(config.SEED + 1, N_MONITOR), labels)
        pred, feats = self.predict(self.clf, images)
        out = {"fid_cnn": fid(gaussian_stats(feats), self.real_stats), "class_entropy": class_entropy(pred)}
        if conditional:
            out["control_acc"] = (pred == labels).float().mean().item()
        return out


def train(kind: str, epochs: int, frames_dir, log=print, seed: int = config.SEED):
    """Entrena un modelo y devuelve (G, D, historial por época, pérdidas por iteración, segundos).

    `seed` fija la inicialización de los pesos, el ruido de entrenamiento y el orden de los lotes. El modelo
    final usa config.SEED (42), fijada de antemano; seeds.py usa otras semillas solo para medir la robustez.
    El ruido fijo de las grillas y el del monitoreo NO dependen de `seed`: así se comparan semillas con las
    mismas entradas.
    """
    config.set_seed(seed)
    conditional = kind == "cgan"
    X, y = load_fashion_mnist(train=True)
    G, D = build(kind)
    log(f"{NAMES[kind]}: G {count_params(G):,} parámetros · D {count_params(D):,} parámetros")
    bce = nn.BCEWithLogitsLoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=HPARAMS["lr"], betas=HPARAMS["betas"])
    opt_D = torch.optim.Adam(D.parameters(), lr=HPARAMS["lr"], betas=HPARAMS["betas"])
    order = torch.Generator().manual_seed(seed)               # orden de los lotes reproducible
    fixed_z, fixed_y = fixed_inputs(kind)
    monitor = Monitor()
    history, iter_losses = [], []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        G.train(), D.train()
        sums = np.zeros(5)  # loss_D, loss_G, D(x), D(G(z)) antes del paso de D, D(G(z)) en el paso de G
        n_batches = 0
        for idx in torch.randperm(len(X), generator=order).split(HPARAMS["batch_size"]):
            real, labels = X[idx], y[idx]
            ones, zeros = torch.ones(len(idx)), torch.zeros(len(idx))
            z = torch.randn(len(idx), NZ)
            # cGAN: las falsas se piden con las mismas etiquetas del lote real (misma mezcla de clases).
            fake = G(z, labels) if conditional else G(z)
            judge = (lambda img: D(img, labels)) if conditional else D

            # 1) Discriminador: maximizar log D(x) + log(1 − D(G(z)))  <=>  minimizar BCE(real,1) + BCE(falsa,0)
            opt_D.zero_grad()
            d_real = judge(real)
            d_fake = judge(fake.detach())        # detach: este paso NO propaga gradientes hacia G
            loss_D = bce(d_real, ones) + bce(d_fake, zeros)
            loss_D.backward()
            opt_D.step()

            # 2) Generador (no saturante): minimizar BCE(D(G(z)), 1) = −log D(G(z)), con el D ya actualizado
            opt_G.zero_grad()
            d_fake_for_G = judge(fake)
            loss_G = bce(d_fake_for_G, ones)
            loss_G.backward()
            opt_G.step()

            sums += [loss_D.item(), loss_G.item(), torch.sigmoid(d_real).mean().item(),
                     torch.sigmoid(d_fake).mean().item(), torch.sigmoid(d_fake_for_G).mean().item()]
            iter_losses.append((loss_D.item(), loss_G.item()))
            n_batches += 1

        row = dict(zip(["loss_D", "loss_G", "D_x", "D_G_z_before", "D_G_z_after"], sums / n_batches))
        row = {"epoch": epoch, **row, **monitor(G, conditional), "seconds": time.perf_counter() - start}
        history.append(row)
        log("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        save_grid(generate(G, fixed_z, fixed_y), frames_dir / f"epoch_{epoch:02d}.png", nrow=8,
                  title=f"{NAMES[kind]} · época {epoch} (ruido fijo)",
                  row_labels=CLASSES if conditional else None)
    return G, D, pd.DataFrame(history), np.array(iter_losses), time.perf_counter() - start


def save_gif(frames_dir, path, ms: int = 450) -> None:
    """Animación del progreso con las grillas de cada época (la última se queda más tiempo en pantalla)."""
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in sorted(frames_dir.glob("epoch_*.png"))]
    durations = [ms] * (len(frames) - 1) + [ms * 5]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def plot_curves(history: pd.DataFrame, iter_losses: np.ndarray, path, title: str, mark_epoch: int | None = None):
    """Pérdidas de D y G, medias de D(x) y D(G(z)) y calidad (FID-CNN), en paneles separados (un eje cada uno).

    `mark_epoch` dibuja una línea vertical en la época elegida (curvas de la exploración)."""
    has_fid = "fid_cnn" in history
    fig, axes = plt.subplots(1, 3 if has_fid else 2, figsize=(15 if has_fid else 10, 4.2))
    per_epoch = len(iter_losses) / len(history)
    it_epochs = np.arange(1, len(iter_losses) + 1) / per_epoch
    smooth = lambda v: pd.Series(v).rolling(50, min_periods=1).mean()
    ax = axes[0]
    for col, key, label in ((0, "D", "Discriminador"), (1, "G", "Generador")):
        ax.plot(it_epochs, iter_losses[:, col], color=COLORS[key], alpha=0.15, lw=0.6)
        ax.plot(it_epochs, smooth(iter_losses[:, col]), color=COLORS[key], lw=2, label=f"{label} (media móvil 50 it.)")
    ax.set(title="Pérdidas (BCE) por iteración", xlabel="época", ylabel="pérdida")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(history["epoch"], history["D_x"], "o-", color=COLORS["D"], lw=2, ms=4, label="D(x): reales")
    ax.plot(history["epoch"], history["D_G_z_before"], "s-", color=COLORS["G"], lw=2, ms=4, label="D(G(z)): falsas")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(1, 0.51, "0,5 = D no distingue", ha="left", va="bottom", fontsize=8, color="gray")
    ax.set(title="Salida media de D (probabilidad de 'real')", xlabel="época", ylim=(0, 1))
    ax.legend(frameon=False, loc="lower right")   # zona libre: D(G(z)) está entre 0,3 y 0,45
    if has_fid:
        ax = axes[2]
        ax.plot(history["epoch"], history["fid_cnn"], "o-", color="#333333", lw=2, ms=4)
        best = history.loc[history["fid_cnn"].idxmin()]
        ax.annotate(f"mínimo {best['fid_cnn']:.2f} (época {int(best['epoch'])})", (best["epoch"], best["fid_cnn"]),
                    xytext=(0, 18), textcoords="offset points", ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="gray"))
        if mark_epoch and mark_epoch <= len(history):
            chosen = history.loc[history["epoch"] == mark_epoch, "fid_cnn"].iloc[0]
            ax.text(mark_epoch, ax.get_ylim()[1] * 0.92, f" elegida: época {mark_epoch}\n FID-CNN {chosen:.1f}",
                    fontsize=8, color="gray", va="top")
        ax.set(title="Calidad: FID-CNN vs. reales de prueba (menor es mejor)", xlabel="época", ylabel="FID-CNN")
        ax.set_ylim(bottom=0)
    for ax in axes:
        if mark_epoch:
            ax.axvline(mark_epoch, color="gray", ls="--", lw=1)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(kind: str, epochs: int, explore: bool = False) -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{kind}_explore" if explore else kind
    frames_dir = (SNAPSHOT_DIR if explore else TRAINING_DIR) / f"{prefix}_epochs"
    for old in frames_dir.glob("epoch_*.png"):     # que no queden grillas de una corrida anterior más larga
        old.unlink()
    G, D, history, iter_losses, seconds = train(kind, epochs, frames_dir)
    history.to_csv(TRAINING_DIR / f"{prefix}_history.csv", index=False, float_format="%.4f")
    plot_curves(history, iter_losses, TRAINING_DIR / f"{prefix}_curves.png",
                f"{NAMES[kind]} · {epochs} épocas · {seconds / 60:.1f} min en CPU" + (" · exploración" if explore else ""),
                mark_epoch=EPOCHS[kind] if explore else None)
    print(f"{NAMES[kind]}: {epochs} épocas en {seconds:.0f} s ({seconds / epochs:.1f} s/época)")
    if explore:
        return
    save_gif(frames_dir, TRAINING_DIR / f"{kind}_progress.gif")
    save_checkpoint(CKPTS[kind], G, D, epochs=epochs, seed=config.SEED, classes=CLASSES,
                    hparams=HPARAMS, train_seconds=round(seconds, 1), final_metrics=history.iloc[-1].to_dict())
    # El checkpoint recargado debe generar EXACTAMENTE lo mismo con la misma semilla.
    G2, _ = load_generator(CKPTS[kind])
    labels = torch.arange(16) % len(CLASSES) if kind == "cgan" else None
    same = torch.equal(generate_from_seed(G, 123, 16, labels), generate_from_seed(G2, 123, 16, labels))
    print(f"checkpoint -> {CKPTS[kind]} · recargado genera las mismas imágenes: {same}")
    if not same:
        raise RuntimeError("el checkpoint recargado no reproduce las imágenes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["dcgan", "cgan", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None, help="por defecto, las de EPOCHS")
    parser.add_argument("--explore", type=int, default=None, metavar="N",
                        help="entrena N épocas solo para elegir cuántas usar (no guarda modelos)")
    args = parser.parse_args()
    for kind in (["dcgan", "cgan"] if args.model == "both" else [args.model]):
        if args.explore:
            run(kind, args.explore, explore=True)
        else:
            run(kind, args.epochs or EPOCHS[kind])


if __name__ == "__main__":
    main()
