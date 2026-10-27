"""Tarea D (Elizabeth): robustez entre semillas. ¿El resultado depende de una semilla afortunada?

Se entrenan la DCGAN y la cGAN con 5 semillas DISTINTAS de la final (1-5), con la misma configuración y las
mismas 20 épocas, y se evalúan exactamente igual que el modelo final (mismas 10 000 muestras de evaluación,
misma CNN auxiliar). El modelo de models/ es el de SEED = 42, fijada de antemano en common/config.py y NO
elegido por ser el mejor: aquí se comprueba dónde cae frente a la distribución de las otras 5 semillas.
(Si 42 estuviera entre las 5, caería en su propio rango por definición y la comprobación no diría nada.)

Los pesos de cada semilla son solo para evaluar: se guardan en checkpoints/task_d/seeds/ (fuera de Git) y
nunca reemplazan los checkpoints finales.

Uso: python -m task_d_gan.seeds              # entrena las semillas que falten (~65 min en CPU) y evalúa
     python -m task_d_gan.seeds --eval-only  # solo evalúa los pesos ya entrenados
Salidas: outputs/task_d/evaluation/seed_results.csv, seed_metrics.png y seed_samples_{dcgan,cgan}.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import config
from task_d_gan.classifier import load_classifier, predict
from task_d_gan.data import CLASSES, load_fashion_mnist, save_grid
from task_d_gan.evaluate import N_GEN, SEED_EVAL, confident_share
from task_d_gan.metrics import (
    class_entropy,
    class_histogram,
    fid,
    gaussian_stats,
    mean_pairwise_distance,
)
from task_d_gan.models import generate, load_generator, noise, save_checkpoint
from task_d_gan.paths import EVAL_JSON, EVALUATION_DIR, SNAPSHOT_DIR
from task_d_gan.train import CKPTS, EPOCHS, HPARAMS, NAMES, train

SEEDS = [1, 2, 3, 4, 5]              # semillas del estudio (todas distintas de config.SEED = 42)
SEED_DIR = SNAPSHOT_DIR / "seeds"
RESULTS_CSV = EVALUATION_DIR / "seed_results.csv"
# Métricas escalares que se comparan entre semillas (columna, nombre legible, ¿mayor es mejor?).
METRICS = {
    "dcgan": [("fid_cnn", "FID-CNN", False), ("classes_covered", "Clases cubiertas (≥ 2 %)", True),
              ("class_entropy", "Entropía de clases", True), ("min_class_share_pct", "Clase menos generada (%)", True),
              ("diversity_l2", "Diversidad (L2 por pares)", True), ("confident_pct", "Reconocibles (%)", True)],
    "cgan": [("fid_cnn", "FID-CNN", False), ("control_pct", "Control global (%)", True),
             ("intra_class_ratio", "Diversidad dentro de clase / real", True),
             ("diversity_l2", "Diversidad (L2 por pares)", True), ("confident_pct", "Reconocibles (%)", True)],
}


def seed_ckpt(kind: str, seed: int):
    return CKPTS[kind] if seed == config.SEED else SEED_DIR / f"{kind}_seed{seed}.pt"


def train_missing(kind: str) -> None:
    for seed in SEEDS:
        path = seed_ckpt(kind, seed)
        if path.exists():
            continue
        print(f"--- {NAMES[kind]} · semilla {seed}")
        G, D, history, _, seconds = train(kind, EPOCHS[kind], SEED_DIR / f"{kind}_seed{seed}_epochs", seed=seed)
        save_checkpoint(path, G, D, epochs=EPOCHS[kind], seed=seed, classes=CLASSES, hparams=HPARAMS,
                        train_seconds=round(seconds, 1), final_metrics=history.iloc[-1].to_dict())
        print(f"{NAMES[kind]} semilla {seed}: {seconds:.0f} s")


@torch.no_grad()
def model_metrics(G, conditional: bool, clf, real_stats, Xt, yt) -> dict:
    """Las mismas métricas de evaluate.py, sobre las mismas 10 000 muestras de evaluación (semilla 142)."""
    labels = torch.arange(N_GEN) % len(CLASSES) if conditional else None
    images = generate(G, noise(SEED_EVAL, N_GEN), labels)
    pred, feats = predict(clf, images)
    out = {"fid_cnn": fid(gaussian_stats(feats), real_stats), "diversity_l2": mean_pairwise_distance(images),
           "confident_pct": confident_share(clf, images), "class_entropy": class_entropy(pred)}
    if conditional:
        out["control_pct"] = float((pred == labels).float().mean()) * 100   # mismo orden de operaciones que evaluate.py
        ratios = []
        for c, name in enumerate(CLASSES):
            mine = labels == c
            out[f"control_{name}"] = float((pred[mine] == c).float().mean() * 100)
            ratios.append(mean_pairwise_distance(images[mine], 5000) / mean_pairwise_distance(Xt[yt == c], 5000))
        out["intra_class_ratio"] = float(np.mean(ratios))
    else:
        hist = class_histogram(pred)
        out.update({"classes_covered": int((hist >= 0.02).sum()), "min_class_share_pct": float(hist.min() * 100),
                    "max_class_share_pct": float(hist.max() * 100)})
    return out


def check_against_summary(df: pd.DataFrame) -> None:
    """Los números de la semilla 42 deben ser los mismos que reporta evaluate.py (mismo código, mismas muestras)."""
    if not EVAL_JSON.exists():
        return
    s = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    final = df[df["seed"] == config.SEED].set_index("model")
    pairs = [(final.loc["dcgan", "fid_cnn"], s["fid_cnn"]["dcgan_vs_real_test"]),
             (final.loc["cgan", "fid_cnn"], s["fid_cnn"]["cgan_vs_real_test"]),
             (final.loc["cgan", "control_pct"], s["cgan"]["control_accuracy"] * 100),
             (final.loc["dcgan", "diversity_l2"], s["diversity_pairwise_l2"]["dcgan"])]
    ok = all(abs(a - b) < 1e-6 for a, b in pairs)
    print("semilla 42 = summary.json de evaluate.py:", ok)
    if not ok:
        raise RuntimeError(f"las métricas de la semilla 42 no coinciden con summary.json: {pairs}")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Filas extra por modelo: media y desviación estándar (n − 1) de las 5 semillas, sin la 42."""
    rows = []
    for kind, block in df.groupby("model", sort=False):
        others = block[block["seed"] != config.SEED].drop(columns=["model", "seed", "final"])
        rows.append({"model": kind, "seed": "media (5 semillas)", **others.mean().to_dict()})
        rows.append({"model": kind, "seed": "desv. est. (5 semillas)", **others.std(ddof=1).to_dict()})
    return pd.DataFrame(rows)


def where_final_falls(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada métrica: valor de la semilla 42, rango [mín, máx] de las otras 5, z y puesto entre las 6."""
    out = []
    for kind, metrics in METRICS.items():
        block = df[df["model"] == kind]
        others = block[block["seed"] != config.SEED]
        final = block[block["seed"] == config.SEED].iloc[0]
        for col, label, higher_better in metrics:
            v, vals = final[col], others[col].to_numpy(float)
            std = vals.std(ddof=1)
            rank = 1 + int((vals > v).sum() if higher_better else (vals < v).sum())  # 1 = el mejor de los 6
            out.append({"model": kind, "metric": label, "seed_42": v, "min_5": vals.min(), "max_5": vals.max(),
                        "mean_5": vals.mean(), "std_5": std, "z_42": (v - vals.mean()) / std if std > 0 else 0.0,
                        "in_range": vals.min() <= v <= vals.max(), "rank_42_of_6": rank})
    return pd.DataFrame(out)


def plot_metrics(df: pd.DataFrame, path) -> None:
    """Un panel por métrica (cada uno con su eje): las 5 semillas en gris y la 42 (modelo final) en color."""
    panels = [(k, m) for k, ms in METRICS.items() for m in ms if m[0] not in ("classes_covered",)]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
    for ax, (kind, (col, label, higher_better)) in zip(axes.flat, panels):
        block = df[df["model"] == kind]
        others = block[block["seed"] != config.SEED]
        final = block[block["seed"] == config.SEED][col].iloc[0]
        ax.scatter(np.zeros(len(others)), others[col], s=46, color="#8A8A8A", zorder=2, label="semillas 1-5")
        for _, r in others.iterrows():
            ax.annotate(str(r["seed"]), (0, r[col]), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8, color="#666666")
        ax.scatter([0], [final], s=90, color="#D9480F", marker="D", zorder=3, label="semilla 42 (final)")
        ax.set_title(f"{NAMES[kind]} · {label}\n({'mayor' if higher_better else 'menor'} es mejor)", fontsize=10)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.grid(alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle("Robustez entre semillas: el modelo final (semilla 42) frente a 5 entrenamientos con otras semillas")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def seed_grids(models: dict) -> None:
    """Mismo ruido fijo para todas las semillas: fila = semilla del entrenamiento."""
    seeds = [config.SEED] + SEEDS
    rows = [f"{s} (final)" if s == config.SEED else f"semilla {s}" for s in seeds]
    z = noise(config.SEED, 10)
    save_grid(torch.cat([generate(models["dcgan"][s], z) for s in seeds]), EVALUATION_DIR / "seed_samples_dcgan.png",
              nrow=10, row_labels=rows, title="DCGAN: el mismo ruido fijo z en cada semilla de entrenamiento")
    z1 = noise(config.SEED, 1).repeat(len(CLASSES), 1)               # un mismo z para las 10 clases
    save_grid(torch.cat([generate(models["cgan"][s], z1, torch.arange(len(CLASSES))) for s in seeds]),
              EVALUATION_DIR / "seed_samples_cgan.png", nrow=10, row_labels=rows, col_labels=CLASSES,
              title="cGAN: las 10 clases con el mismo z, en cada semilla de entrenamiento")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-only", action="store_true", help="no entrena; evalúa los pesos ya guardados")
    args = parser.parse_args()
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        for kind in ("dcgan", "cgan"):
            train_missing(kind)

    clf = load_classifier()
    Xt, yt = load_fashion_mnist(train=False)
    real_stats = gaussian_stats(predict(clf, Xt)[1])
    rows, models = [], {"dcgan": {}, "cgan": {}}
    for kind in ("dcgan", "cgan"):
        for seed in [config.SEED] + SEEDS:
            G, ckpt = load_generator(seed_ckpt(kind, seed))
            models[kind][seed] = G
            m = model_metrics(G, kind == "cgan", clf, real_stats, Xt, yt)
            rows.append({"model": kind, "seed": seed, "final": seed == config.SEED,
                         "train_seconds": ckpt["train_seconds"], **m})
            print(f"{NAMES[kind]} semilla {seed}: FID-CNN {m['fid_cnn']:.2f}"
                  + (f" · control {m['control_pct']:.1f} %" if kind == "cgan" else
                     f" · {m['classes_covered']}/10 clases"))
    df = pd.DataFrame(rows)
    check_against_summary(df)
    pd.concat([df, summarize(df)], ignore_index=True).to_csv(RESULTS_CSV, index=False, float_format="%.4f")
    placement = where_final_falls(df)
    placement.to_csv(EVALUATION_DIR / "seed_final_vs_others.csv", index=False, float_format="%.4f")
    print(placement.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plot_metrics(df, EVALUATION_DIR / "seed_metrics.png")
    seed_grids(models)
    print("->", RESULTS_CSV)


if __name__ == "__main__":
    main()
