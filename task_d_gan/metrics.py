"""Tarea D (Elizabeth): métricas para juzgar una GAN con evidencia, no solo a ojo.

- fid(): distancia de Fréchet entre dos nubes de rasgos (FID). Aquí los rasgos salen de NUESTRA CNN auxiliar
  (128 valores de la penúltima capa), no de Inception-v3 como en el FID estándar: por eso lo llamamos
  "FID-CNN". Solo sirve para COMPARAR modelos y épocas entre sí con el mismo extractor; sus valores no son
  comparables con FID publicados.
- class_entropy(): qué tan pareja es la mezcla de clases (1 = las 10 clases por igual; 0 = una sola clase).
- mean_pairwise_distance(): diversidad (distancia L2 media entre pares de imágenes).
- nearest_neighbors(): para cada imagen, la más cercana (L2) de un conjunto (memorización).
"""
import warnings

import numpy as np
import torch
from scipy import linalg


def gaussian_stats(feats: torch.Tensor):
    f = feats.double().cpu().numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def fid(stats_a, stats_b) -> float:
    """||mu_a - mu_b||² + Tr(S_a + S_b - 2·(S_a·S_b)^½)."""
    (mu1, s1), (mu2, s2) = stats_a, stats_b
    # Algunas neuronas ReLU de la CNN valen siempre 0 -> covarianza singular. sqrtm sigue siendo válida (se toma
    # la parte real); solo se silencia el aviso.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():  # estabilización numérica habitual
        eps = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def class_histogram(pred: torch.Tensor, n_classes: int = 10) -> np.ndarray:
    """Fracción de imágenes predichas en cada clase."""
    return torch.bincount(pred, minlength=n_classes).double().div(len(pred)).numpy()


def class_entropy(pred: torch.Tensor, n_classes: int = 10) -> float:
    """Entropía de la mezcla de clases normalizada a [0, 1] (1 = uniforme)."""
    p = class_histogram(pred, n_classes)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(n_classes))


def mean_pairwise_distance(X: torch.Tensor, n_pairs: int = 20000, seed: int = 0) -> float:
    """Distancia L2 media entre pares aleatorios de imágenes distintas (píxeles en [-1, 1]).

    Si una GAN colapsa, sus muestras se parecen entre sí y esta distancia cae muy por debajo de la de las reales.
    """
    g = torch.Generator().manual_seed(seed)
    flat = X.reshape(len(X), -1)
    i = torch.randint(len(X), (n_pairs,), generator=g)
    j = torch.randint(len(X) - 1, (n_pairs,), generator=g)
    j = j + (j >= i).long()                                   # nunca el mismo índice
    return float((flat[i] - flat[j]).norm(dim=1).mean())


@torch.no_grad()
def nearest_neighbors(queries: torch.Tensor, pool: torch.Tensor, batch: int = 256):
    """(distancia L2, índice) de la imagen de `pool` más cercana a cada imagen de `queries`."""
    q = queries.reshape(len(queries), -1)
    p = pool.reshape(len(pool), -1)
    p_sq = (p * p).sum(1)
    dists, idxs = [], []
    for i in range(0, len(q), batch):
        qb = q[i:i + batch]
        # ||a - b||² = ||a||² + ||b||² - 2 a·b  (sin construir todas las diferencias: mucho más rápido)
        d2 = (qb * qb).sum(1, keepdim=True) + p_sq.unsqueeze(0) - 2 * qb @ p.T
        d, k = d2.clamp_min(0).min(1)
        dists.append(d.sqrt())
        idxs.append(k)
    return torch.cat(dists), torch.cat(idxs)
