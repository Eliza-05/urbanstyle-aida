"""Tarea D (Elizabeth): evaluación de la DCGAN y la cGAN con evidencia medible (no solo a ojo).

a) CNN auxiliar: exactitud en test (el "juez" debe ser confiable).
b) DCGAN: cobertura de clases de 10 000 muestras (colapso de modos) y diversidad (distancia media por pares).
c) cGAN: matriz clase pedida vs. clase predicha (¿la etiqueta controla la prenda?) y diversidad DENTRO de cada clase.
d) Memorización: vecino más cercano (L2) en las 60 000 de entrenamiento, comparado con el de imágenes reales nuevas.
e) FID-CNN: reales vs. DCGAN vs. cGAN (y reales vs. reales como piso).
f) Interpolación latente (hito 5 del taller): DCGAN, y cGAN con la clase fija.

Uso: python -m task_d_gan.evaluate   (~1 min; requiere models/gan_generator.pt, cgan_generator.pt y fashion_cnn.pt)
Salidas: outputs/task_d/evaluation/ (figuras + summary.json) y outputs/task_d/samples/
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import config
from task_d_gan.classifier import load_classifier, predict
from task_d_gan.data import CLASSES, examples_per_class, load_fashion_mnist, save_grid
from task_d_gan.metrics import (
    class_entropy,
    class_histogram,
    fid,
    gaussian_stats,
    mean_pairwise_distance,
    nearest_neighbors,
)
from task_d_gan.models import generate, interpolate, load_generator, noise
from task_d_gan.paths import CGAN_CKPT, EVAL_JSON, EVALUATION_DIR, GAN_CKPT, SAMPLES_DIR

N_GEN = 10_000           # muestras para cobertura, FID-CNN y diversidad
N_NN = 1_000             # muestras para la búsqueda del vecino más cercano
CONFIDENT = 0.9          # umbral de "la CNN reconoce la prenda sin dudar"
SEED_EVAL = config.SEED + 100   # ruido distinto del usado para monitorear el entrenamiento
COLORS = {"real": "#8A8A8A", "dcgan": "#D9480F", "cgan": "#2B6CB0"}


def style(ax) -> None:
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / name, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# b) Cobertura de clases y diversidad
# ---------------------------------------------------------------------------

def plot_coverage(real_hist, gen_hist, path_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(CLASSES))
    ax.bar(x - 0.2, real_hist * 100, 0.38, color=COLORS["real"], label="Reales de prueba (lo que la CNN predice)")
    ax.bar(x + 0.2, gen_hist * 100, 0.38, color=COLORS["dcgan"], label="DCGAN (10 000 muestras)")
    ax.axhline(10, color="black", lw=0.8, ls=":", label="10 % = reparto perfecto")
    ax.set_xticks(x, CLASSES)
    ax.set(ylabel="% de imágenes", ylim=(0, max(real_hist.max(), gen_hist.max()) * 100 * 1.3),
           title="DCGAN: ¿qué prendas genera? Clase predicha por la CNN auxiliar")
    ax.legend(frameon=False, loc="upper center", ncol=3)
    style(ax)
    save(fig, path_name)


def by_predicted_class(images, pred, per_class: int = 10) -> torch.Tensor:
    """Grilla 'filas = clase predicha' (si una clase tiene menos muestras se rellena con negro)."""
    rows = []
    for c in range(len(CLASSES)):
        sel = images[pred == c][:per_class]
        pad = torch.full((per_class - len(sel), 1, 28, 28), -1.0)
        rows.append(torch.cat([sel, pad]))
    return torch.cat(rows)


@torch.no_grad()
def confident_share(clf, images: torch.Tensor) -> float:
    """% de imágenes que la CNN clasifica con probabilidad máxima > CONFIDENT: mide qué tan reconocibles son
    (misma idea que el Inception Score). Las muestras ambiguas o híbridas entre clases quedan por debajo."""
    clf.eval()
    probs = torch.cat([torch.softmax(clf(b), 1) for b in images.split(1000)])
    return float((probs.max(1).values > CONFIDENT).float().mean() * 100)


# ---------------------------------------------------------------------------
# c) Control de la cGAN
# ---------------------------------------------------------------------------

def confusion(requested: torch.Tensor, pred: torch.Tensor, n: int = 10) -> np.ndarray:
    """Matriz n×n en % por fila: fila = clase pedida (o real), columna = clase predicha."""
    m = np.zeros((n, n))
    for r, p in zip(requested.tolist(), pred.tolist()):
        m[r, p] += 1
    return m / m.sum(1, keepdims=True) * 100


def plot_confusion(m: np.ndarray, title: str, ylabel: str, path_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.imshow(m, cmap="Oranges", vmin=0, vmax=100)
    for i in range(len(m)):
        for j in range(len(m)):
            if m[i, j] >= 0.5:
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 55 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Clase que predice la CNN auxiliar", ylabel=ylabel, title=title)
    save(fig, path_name)


def plot_control_vs_ceiling(cgan_diag, real_diag, path_name: str) -> None:
    """Por clase: % de la cGAN clasificado como la clase pedida vs. exactitud de la CNN en reales (referencia: la
    medición no puede ser mucho mejor que el propio juez)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(CLASSES))
    ax.bar(x - 0.2, real_diag, 0.38, color=COLORS["real"], label="CNN en imágenes reales (referencia del juez)")
    ax.bar(x + 0.2, cgan_diag, 0.38, color=COLORS["cgan"], label="cGAN: % clasificado como la clase pedida")
    ax.set_xticks(x, CLASSES)
    ax.set(ylabel="%", ylim=(0, 125), yticks=range(0, 101, 20), title="cGAN: la clase pedida controla la prenda generada")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    style(ax)
    save(fig, path_name)


# ---------------------------------------------------------------------------
# d) Memorización
# ---------------------------------------------------------------------------

def plot_nn_pairs(gen_dcgan, nn_dcgan, gen_cgan, nn_cgan, d_dcgan, d_cgan, path_name: str) -> None:
    """Generada (arriba) y su imagen de entrenamiento más cercana (abajo), con la distancia L2."""
    k = len(gen_dcgan)
    fig, axes = plt.subplots(4, k, figsize=(1.25 * k, 6.2))
    rows = [(gen_dcgan, "DCGAN\ngenerada", d_dcgan), (nn_dcgan, "más cercana\nen train", None),
            (gen_cgan, "cGAN\ngenerada", d_cgan), (nn_cgan, "más cercana\nen train", None)]
    for r, (imgs, label, dist) in enumerate(rows):
        for c in range(k):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0].numpy(), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            if dist is not None:
                ax.set_title(f"L2 = {dist[c]:.1f}", fontsize=7, pad=2)
    fig.suptitle("Memorización: ¿la GAN copia imágenes de entrenamiento? (vecino más cercano en 60 000)")
    save(fig, path_name)


def plot_nn_hist(d_test, d_dcgan, d_cgan, path_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(d_test.max(), d_dcgan.max(), d_cgan.max()), 50)
    for d, key, label in ((d_test, "real", "Reales NUEVAS (prueba): referencia de 'no copia'"),
                          (d_dcgan, "dcgan", "DCGAN"), (d_cgan, "cgan", "cGAN")):
        ax.hist(d, bins, histtype="step", lw=2, color=COLORS[key], label=f"{label} · mediana {np.median(d):.1f}")
    ax.set(xlabel="Distancia L2 a la imagen de entrenamiento más cercana (píxeles en [-1, 1])",
           ylabel="imágenes", title="Distancia al vecino más cercano en entrenamiento (1000 imágenes por grupo)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.4)          # espacio para la leyenda sin tapar las barras
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    save(fig, path_name)


# ---------------------------------------------------------------------------
# f) Interpolación latente
# ---------------------------------------------------------------------------

def interpolation_grid(G, pairs, steps: int = 10, labels=None) -> torch.Tensor:
    rows = []
    for i, (s0, s1) in enumerate(pairs):
        z = interpolate(noise(s0, 1)[0], noise(s1, 1)[0], steps)
        rows.append(generate(G, z, None if labels is None else labels[i]))
    return torch.cat(rows)


# ---------------------------------------------------------------------------

def main() -> None:
    config.set_seed()
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_fashion_mnist(train=True)
    Xt, yt = load_fashion_mnist(train=False)
    clf = load_classifier()
    G, g_ckpt = load_generator(GAN_CKPT)
    cG, c_ckpt = load_generator(CGAN_CKPT)
    summary = {}

    # a) Juez confiable
    pred_t, feat_t = predict(clf, Xt)
    real_conf = confusion(yt, pred_t)
    summary["classifier"] = {"test_accuracy": float((pred_t == yt).float().mean()),
                             "per_class": dict(zip(CLASSES, np.diag(real_conf).round(1).tolist()))}
    print("a) CNN auxiliar, exactitud en test:", round(summary["classifier"]["test_accuracy"], 4))

    # b) DCGAN: cobertura y diversidad
    gen = generate(G, noise(SEED_EVAL, N_GEN))
    pred_g, feat_g = predict(clf, gen)
    real_hist, gen_hist = class_histogram(pred_t), class_histogram(pred_g)
    plot_coverage(real_hist, gen_hist, "dcgan_class_coverage.png")
    save_grid(by_predicted_class(gen, pred_g), SAMPLES_DIR / "dcgan_by_predicted_class.png", nrow=10,
              row_labels=CLASSES, title="DCGAN: muestras agrupadas por la clase que predice la CNN")
    save_grid(gen[:100], SAMPLES_DIR / "dcgan_samples.png", nrow=10,
              title=f"DCGAN: 100 muestras al azar (semilla {SEED_EVAL})")
    diversity = {"real_test": mean_pairwise_distance(Xt), "dcgan": mean_pairwise_distance(gen)}
    summary["dcgan"] = {
        "n_samples": N_GEN, "epochs": g_ckpt["epochs"], "train_seconds": g_ckpt["train_seconds"],
        "class_share_pct": dict(zip(CLASSES, (gen_hist * 100).round(1).tolist())),
        "real_class_share_pct": dict(zip(CLASSES, (real_hist * 100).round(1).tolist())),
        "classes_covered": int((gen_hist >= 0.02).sum()),           # clases con al menos 2 % de las muestras
        "min_class_share_pct": float(gen_hist.min() * 100), "max_class_share_pct": float(gen_hist.max() * 100),
        "class_entropy": class_entropy(pred_g), "real_class_entropy": class_entropy(pred_t),
        "total_variation_vs_real": float(0.5 * np.abs(gen_hist - real_hist).sum()),
    }
    print(f"b) DCGAN cubre {summary['dcgan']['classes_covered']}/10 clases · entropía "
          f"{summary['dcgan']['class_entropy']:.3f} (reales {summary['dcgan']['real_class_entropy']:.3f})")

    # c) cGAN: pedida vs. predicha (1000 por clase) y diversidad dentro de cada clase
    requested = torch.arange(N_GEN) % len(CLASSES)
    cgen = generate(cG, noise(SEED_EVAL, N_GEN), requested)
    pred_c, feat_c = predict(clf, cgen)
    m = confusion(requested, pred_c)
    plot_confusion(m, "cGAN: clase pedida vs. clase predicha (% por fila, 1000 por clase)",
                   "Clase pedida a la cGAN", "cgan_confusion.png")
    plot_confusion(real_conf, "Referencia: CNN auxiliar en reales de prueba (% por fila)", "Clase real",
                   "classifier_confusion_real.png")
    plot_control_vs_ceiling(np.diag(m), np.diag(real_conf), "cgan_control_by_class.png")
    rows = torch.cat([cgen[requested == c][:10] for c in range(len(CLASSES))])
    save_grid(rows, SAMPLES_DIR / "cgan_by_class.png", nrow=10, row_labels=CLASSES,
              title="cGAN: 10 muestras por clase pedida")
    diversity["cgan"] = mean_pairwise_distance(cgen)
    intra = {name: {"real": mean_pairwise_distance(Xt[yt == c], 5000),
                    "cgan": mean_pairwise_distance(cgen[requested == c], 5000)} for c, name in enumerate(CLASSES)}
    summary["cgan"] = {
        "n_samples": N_GEN, "epochs": c_ckpt["epochs"], "train_seconds": c_ckpt["train_seconds"],
        "control_accuracy": float((pred_c == requested).float().mean()),
        "control_by_class_pct": dict(zip(CLASSES, np.diag(m).round(1).tolist())),
        "intra_class_diversity": intra,
        "intra_class_ratio_mean": float(np.mean([v["cgan"] / v["real"] for v in intra.values()])),
    }
    print(f"c) cGAN: {summary['cgan']['control_accuracy']:.3f} de las imágenes son de la clase pedida "
          f"(CNN en reales: {summary['classifier']['test_accuracy']:.3f})")

    # d) Memorización: vecino más cercano en entrenamiento
    d_test, _ = nearest_neighbors(Xt[:N_NN], X)
    d_g, i_g = nearest_neighbors(gen[:N_NN], X)
    d_c, i_c = nearest_neighbors(cgen[:N_NN], X)
    plot_nn_hist(d_test.numpy(), d_g.numpy(), d_c.numpy(), "memorization_nn_distance.png")
    k = 10
    pick_g = torch.argsort(d_g)[:k]          # las MÁS parecidas a algo de train: el peor caso para "no copia"
    pick_c = torch.argsort(d_c)[:k]
    plot_nn_pairs(gen[pick_g], X[i_g[pick_g]], cgen[pick_c], X[i_c[pick_c]], d_g[pick_g], d_c[pick_c],
                  "memorization_nearest_pairs.png")
    q = lambda d: {"median": float(d.median()), "p1": float(d.quantile(0.01)), "min": float(d.min())}
    summary["memorization"] = {"real_test_to_train": q(d_test), "dcgan_to_train": q(d_g), "cgan_to_train": q(d_c),
                               "dcgan_below_test_p1_pct": float((d_g < d_test.quantile(0.01)).float().mean() * 100),
                               "cgan_below_test_p1_pct": float((d_c < d_test.quantile(0.01)).float().mean() * 100)}
    print(f"d) mediana distancia al vecino de train: reales nuevas {q(d_test)['median']:.2f} · "
          f"DCGAN {q(d_g)['median']:.2f} · cGAN {q(d_c)['median']:.2f}")

    # e) FID-CNN (mismo extractor para todos; 10 000 imágenes por grupo)
    test_stats = gaussian_stats(feat_t)
    train_sub = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:N_GEN]]
    summary["fid_cnn"] = {
        "real_train_vs_real_test": fid(gaussian_stats(predict(clf, train_sub)[1]), test_stats),
        "dcgan_vs_real_test": fid(gaussian_stats(feat_g), test_stats),
        "cgan_vs_real_test": fid(gaussian_stats(feat_c), test_stats),
        "noise_vs_real_test": fid(gaussian_stats(predict(clf, torch.rand(N_GEN, 1, 28, 28) * 2 - 1)[1]), test_stats),
    }
    # BatchNorm en modo entrenamiento (estadísticas del lote) vs. eval (guardadas): ¿generar en eval empeora?
    G.train()
    with torch.no_grad():
        gen_bn_train = torch.cat([G(z) for z in noise(SEED_EVAL, N_GEN).split(128)])
    G.eval()
    summary["fid_cnn"]["dcgan_bn_batch_stats_vs_real_test"] = fid(gaussian_stats(predict(clf, gen_bn_train)[1]),
                                                                  test_stats)
    print("e) FID-CNN:", {k: round(v, 2) for k, v in summary["fid_cnn"].items()})
    summary["diversity_pairwise_l2"] = diversity
    summary["confident_pct"] = {"real_test": confident_share(clf, Xt), "dcgan": confident_share(clf, gen),
                                "cgan": confident_share(clf, cgen)}
    print(f"   reconocibles (confianza > {CONFIDENT}):", {k: round(v, 1) for k, v in summary["confident_pct"].items()})
    print("   diversidad (L2 media por pares):", {k: round(v, 2) for k, v in diversity.items()})

    # f) Interpolación latente
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    save_grid(interpolation_grid(G, pairs), SAMPLES_DIR / "dcgan_interpolation.png", nrow=10,
              row_labels=[f"z{a} → z{b}" for a, b in pairs],
              title="DCGAN: interpolación lineal en el espacio latente (10 pasos)")
    classes = [0, 1, 3, 5, 7, 9]
    save_grid(interpolation_grid(cG, pairs, labels=classes), SAMPLES_DIR / "cgan_interpolation.png", nrow=10,
              row_labels=[CLASSES[c] for c in classes],
              title="cGAN: interpolación de z con la clase FIJA en cada fila")
    real_grid = examples_per_class(Xt, yt, 10)
    save_grid(real_grid, SAMPLES_DIR / "real_test_grid.png", nrow=10, row_labels=CLASSES,
              title="Reales de prueba (para comparar con cgan_by_class.png)")

    # Diagnóstico del entrenamiento (de los checkpoints): ¿D dominó al final?
    summary["training_final"] = {"dcgan": g_ckpt["final_metrics"], "cgan": c_ckpt["final_metrics"]}
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("resumen ->", EVAL_JSON)


if __name__ == "__main__":
    main()
