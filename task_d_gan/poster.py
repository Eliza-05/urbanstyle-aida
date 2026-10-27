"""Tarea D (Elizabeth): póster promocional de UrbanStyle compuesto con prendas GENERADAS por la cGAN (bonus 2).

Cómo se arma:
  1. Para cada una de las 10 clases, la cGAN genera CANDIDATES imágenes con la semilla dada. Entre las que la
     CNN auxiliar reconoce como esa clase con confianza > 0,9 se elige el MEDOIDE en el espacio de rasgos de la
     CNN: la candidata más "típica" (menor distancia total a las demás). Se probó elegir la de mayor confianza y
     daba muestras atípicas (lo más distintivo para el clasificador, no lo más representativo).
  2. Cada imagen de 28×28 se amplía con Lanczos (la más nítida sin halos; "nearest" deja escalones) y se "tiñe": el negro
     del fondo pasa al tono del póster y el blanco de la prenda al marfil de la marca, con un ajuste de niveles
     por prenda (solo en el póster: la pantalla muestra la salida del generador sin retocar).
  3. Se compone en el estilo editorial del proyecto (fondo oscuro, titular en DM Serif Display, un acento) con
     una nota VISIBLE de que las prendas son generadas por IA, no fotos de productos reales.
Se dibuja con pygame (sin ventana), así la pantalla de la Tarea D usa exactamente la misma función.

Uso: python -m task_d_gan.poster [--seed 7]   -> outputs/task_d/poster/poster_seed_7.png
"""
import argparse

import numpy as np
import pygame
import torch
from PIL import Image

from common import config
from common.ui import FONT_DIR, FONT_FILES, THEME
from task_d_gan.data import CLASSES, to_uint8
from task_d_gan.models import generate, load_generator, noise
from task_d_gan.paths import CGAN_CKPT, CLASSIFIER_CKPT, POSTER_DIR

SIZE = (1200, 1500)                 # vertical 4:5 (px), formato habitual de redes sociales
CANDIDATES = 16                     # candidatas por prenda entre las que se elige la mejor
MIN_CONFIDENCE = 0.9                # la candidata debe ser reconocible como la clase pedida
HERO = 9                            # protagonista: Botín (silueta sólida, aguanta bien la ampliación)
SIDE = [3, 0, 8, 4]                 # 2×2 junto a la protagonista: Vestido, Camiseta, Bolso, Abrigo
BOTTOM = [1, 2, 5, 6, 7]            # fila inferior: Pantalón, Suéter, Sandalia, Camisa, Tenis
TILE_BG = (24, 22, 20)              # fondo de cada prenda: apenas más claro que el póster
IVORY = THEME.text
DISCLAIMER = "Prendas generadas por IA (cGAN entrenada en Fashion-MNIST)"

_fonts = {}


def _font(size: int, weight: str = "regular") -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    key = (size, weight)
    if key not in _fonts:
        path = FONT_DIR / FONT_FILES[weight]
        _fonts[key] = pygame.font.Font(str(path) if path.exists() else None, size)
    return _fonts[key]


def _text(surface, text, pos, size, color=IVORY, weight="regular", anchor="topleft") -> pygame.Rect:
    img = _font(size, weight).render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def _tracked(surface, text, pos, size, color, gap: int = 3, weight="semibold", anchor="topleft") -> pygame.Rect:
    """Mayúsculas espaciadas (etiqueta de revista)."""
    glyphs = [_font(size, weight).render(ch, True, color) for ch in text.upper()]
    width = sum(g.get_width() for g in glyphs) + gap * (len(glyphs) - 1)
    rect = pygame.Rect(0, 0, width, _font(size, weight).get_height())
    setattr(rect, anchor, pos)
    x = rect.left
    for g in glyphs:
        surface.blit(g, (x, rect.top))
        x += g.get_width() + gap
    return rect


def pick_garments(cG, seed: int, clf=None) -> torch.Tensor:
    """Una imagen por clase (10, 1, 28, 28) en [-1, 1]; misma semilla -> mismo póster."""
    z = noise(seed, CANDIDATES)
    chosen = []
    for c in range(len(CLASSES)):
        imgs = generate(cG, z, c)
        best = 0
        if clf is not None:
            with torch.no_grad():
                feats = clf.features(imgs)
                conf = torch.softmax(clf.head(feats), 1)[:, c]
            ok = conf > MIN_CONFIDENCE
            if not ok.any():                         # ninguna supera el umbral: las más confiables
                ok = conf >= conf.max()
            pool = torch.nonzero(ok).squeeze(1)
            total_dist = torch.cdist(feats[pool], feats[pool]).sum(1)
            best = int(pool[total_dist.argmin()])    # medoide: la más central entre las reconocibles
        chosen.append(imgs[best])
    return torch.stack(chosen)


def garment_surface(image: torch.Tensor, size: int, levels: bool = False) -> pygame.Surface:
    """28×28 en [-1, 1] -> superficie size×size ampliada (Lanczos) y teñida: fondo -> TILE_BG, prenda -> marfil.

    levels=True (solo el póster): ajuste de niveles como en edición de fotos; el percentil 99 del brillo de la
    prenda pasa a blanco, para que las prendas oscuras no se vean apagadas. La pantalla usa levels=False y
    muestra la salida del generador sin retocar.
    """
    gray = Image.fromarray(to_uint8(image.unsqueeze(0))[0])
    t = np.asarray(gray.resize((size, size), Image.LANCZOS), dtype=np.float32)[..., None] / 255.0
    if levels:
        t = np.clip(t / max(np.percentile(t, 99), 0.35), 0, 1)   # tope 0,35: no amplificar el ruido de fondo
    rgb = np.array(TILE_BG, np.float32) * (1 - t) + np.array(IVORY, np.float32) * t
    return pygame.surfarray.make_surface(rgb.clip(0, 255).astype(np.uint8).transpose(1, 0, 2))


def _tile(surface, image, rect: pygame.Rect, number: int, name: str, big: bool = False) -> None:
    surface.blit(garment_surface(image, rect.width, levels=True), rect.topleft)
    y = rect.bottom + (14 if big else 10)
    num = _text(surface, f"{number:02d}", (rect.left, y), 26 if big else 19, THEME.accent, "serif")
    _text(surface, name, (num.right + 10, num.bottom - (5 if big else 4)), 22 if big else 17, IVORY, "medium",
          anchor="bottomleft")


def make_poster(images: torch.Tensor, seed: int) -> pygame.Surface:
    """Compone el póster (pygame.Surface de SIZE) con las 10 prendas generadas (una por clase)."""
    W, H = SIZE
    m = 72
    poster = pygame.Surface(SIZE)
    poster.fill(THEME.bg)

    # Cabecera: etiqueta, titular de campaña y bajada.
    _tracked(poster, "UrbanStyle · Colección otoño 2026", (m, m), 18, THEME.accent_text)
    _text(poster, f"N.º {seed:03d}", (W - m, m - 4), 22, THEME.text_faint, "serif", anchor="topright")
    title = _text(poster, "La colección", (m, m + 40), 104, IVORY, "serif")
    title2 = _text(poster, "que nadie dibujó.", (m, title.bottom - 18), 104, THEME.accent, "serif_italic")
    pygame.draw.rect(poster, THEME.accent, (m, title2.bottom + 18, 72, 4))
    _text(poster, "Diez prendas imaginadas por una red generativa. Ninguna existía antes de este póster.",
          (m, title2.bottom + 40), 22, THEME.text_muted)

    # Protagonista (izquierda) + 2×2 (derecha) + fila inferior de 5.
    gap = 28
    top = title2.bottom + 96
    side, label_h = 234, 52                     # 2×2 a la derecha: dos filas (con su rótulo) = alto de la protagonista
    hero = pygame.Rect(m, top, 2 * side + label_h, 2 * side + label_h)
    _tile(poster, images[HERO], hero, HERO + 1, CLASSES[HERO], big=True)
    side_x = W - m - 2 * side - gap
    for i, c in enumerate(SIDE):
        r, col = divmod(i, 2)
        rect = pygame.Rect(side_x + col * (side + gap), top + r * (side + label_h), side, side)
        _tile(poster, images[c], rect, c + 1, CLASSES[c])
    row_y = hero.bottom + 92
    small = (W - 2 * m - 4 * 20) // 5
    for i, c in enumerate(BOTTOM):
        _tile(poster, images[c], pygame.Rect(m + i * (small + 20), row_y, small, small), c + 1, CLASSES[c])

    # Pie: nota visible de que las prendas son generadas (no reales) + marca.
    foot = H - m - 58
    pygame.draw.rect(poster, (*THEME.text_faint,), (m, foot, W - 2 * m, 1))
    pygame.draw.rect(poster, THEME.accent, (m, foot + 22, 4, 40))
    _text(poster, DISCLAIMER, (m + 18, foot + 20), 22, IVORY, "semibold")
    _text(poster, f"Imágenes de 28×28 px ampliadas · semilla del ruido {seed}", (m + 18, foot + 48), 17,
          THEME.text_muted)
    _text(poster, "UrbanStyle", (W - m, foot + 62), 44, IVORY, "serif", anchor="bottomright")
    return poster


def build_poster(seed: int, cG=None, clf=None) -> pygame.Surface:
    """Carga lo que falte (cGAN y CNN auxiliar), elige las prendas y compone el póster."""
    if cG is None:
        cG, _ = load_generator(CGAN_CKPT)
    if clf is None and CLASSIFIER_CKPT.exists():
        from task_d_gan.classifier import load_classifier
        clf = load_classifier()
    return make_poster(pick_garments(cG, seed, clf), seed)


def save_poster(poster: pygame.Surface, seed: int):
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    path = POSTER_DIR / f"poster_seed_{seed}.png"
    pygame.image.save(poster, str(path))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Póster promocional con prendas generadas por la cGAN")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    path = save_poster(build_poster(args.seed), args.seed)
    print("póster ->", path)


if __name__ == "__main__":
    main()
