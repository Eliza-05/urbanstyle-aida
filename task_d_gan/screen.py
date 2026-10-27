"""Tarea D (Elizabeth): pantalla Pygame de la GAN, en el lenguaje editorial de la app (igual que la Tarea B).

Qué se ve: un titular que dice lo que muestra la grilla, la grilla de prendas generadas (protagonista), y a la
derecha los controles (modelo DCGAN/cGAN, semilla del ruido con su slider, botón Generar, prenda de la cGAN) y
la evidencia de la evaluación (cobertura de clases de la DCGAN, control de la cGAN).
Vistas: "Grilla" e "Interpolación" (animación de z entre la semilla s y la s+1, hito 5 del taller).
"Crear póster" compone el póster con la cGAN y lo guarda en PNG.

Solo CARGA los modelos ya entrenados (models/*.pt) y outputs/task_d/evaluation/summary.json; nunca entrena.
La misma semilla siempre produce las mismas imágenes (generación en modo eval, ver models.generate).
Geometría en píxeles lógicos (ventana base 1024×640) convertida con ui.px().
"""
import json
import random

import pygame
import torch

from common import config
from common import ui
from common.ui import THEME, px
from task_d_gan.data import CLASSES

GRID_COLS, GRID_ROWS = 5, 3
N_IMAGES = GRID_COLS * GRID_ROWS
SEED_MAX = 999
INTERP_STEPS = 10           # pasos de la tira (como el hito 5 del taller)
INTERP_FRAMES = 48          # cuadros de la animación (ida y vuelta)
INTERP_MS = 3200            # duración de un ciclo ida-vuelta
PLURALS = ["camisetas", "pantalones", "suéteres", "vestidos", "abrigos",
           "sandalias", "camisas", "tenis", "bolsos", "botines"]
FEMININE = {0, 5, 6}        # camisetas, sandalias, camisas
MODELS = {"dcgan": "DCGAN", "cgan": "cGAN"}
VIEWS = {"grid": "Grilla", "interp": "Interpolación"}

_cache = {}


# ---------------------------------------------------------------------------
# Datos: modelos ya entrenados + resumen de la evaluación
# ---------------------------------------------------------------------------

def load_cached() -> dict:
    """Carga (una vez por sesión) los dos generadores, la CNN auxiliar (para el póster) y la evaluación."""
    if "res" not in _cache:
        from task_d_gan.models import load_generator
        from task_d_gan.paths import CGAN_CKPT, CLASSIFIER_CKPT, EVAL_JSON, GAN_CKPT
        res = {}
        res["dcgan"], res["dcgan_ckpt"] = load_generator(GAN_CKPT)
        res["cgan"], res["cgan_ckpt"] = load_generator(CGAN_CKPT)
        res["clf"] = None
        if CLASSIFIER_CKPT.exists():
            from task_d_gan.classifier import load_classifier
            res["clf"] = load_classifier()
        res["summary"] = json.loads(EVAL_JSON.read_text(encoding="utf-8")) if EVAL_JSON.exists() else None
        _cache["res"] = res
    return _cache["res"]


def headline(model: str, class_idx: int, n: int, view: str, seed: int) -> str:
    """Titular que dice lo que se ve."""
    if view == "interp":
        return f"De la semilla {seed} a la {seed + 1}: la prenda cambia de forma gradual"
    if model == "dcgan":
        return f"{n} prendas nuevas, generadas por la DCGAN"
    ending = "as" if class_idx in FEMININE else "os"   # concordancia: "camisetas nuevas", "abrigos nuevos"
    return f"{n} {PLURALS[class_idx]} nuev{ending}, pedid{ending} a la cGAN"


def fmt_pct(value: float) -> str:
    return f"{value:.0f} %"


def R(x, y, w, h) -> pygame.Rect:
    return pygame.Rect(px(x), px(y), px(w), px(h))


# ---------------------------------------------------------------------------
# Pantalla
# ---------------------------------------------------------------------------

class GanScreen:
    """Estado e interacción. Métodos: handle_event(event) y draw(surface)."""

    def __init__(self, res=None, error: str | None = None, size=None):
        self.size = size or (px(config.WINDOW_WIDTH), px(config.WINDOW_HEIGHT))
        self.res, self.error = res, error
        self.done = False
        self.model, self.view = "dcgan", "grid"
        self.seed, self.class_idx = config.SEED, 3            # Vestido
        self.rng = random.Random(config.SEED)                 # "Generar": semillas nuevas pero reproducibles
        self.mouse = (-1, -1)
        self.reveal_start = -10_000                           # aparición escalonada tras "Generar"
        self.toast, self.fade = ui.Toast(), ui.FadeIn()
        self.poster = None                                    # (superficie a pantalla, ruta) si está abierto
        self.hit = {}                                         # zonas clicables de las palabras-selector
        self._cursor = None
        self.layout()
        self.back_button = ui.Button(self.back_rect, "← Menú")
        if res is None:
            return
        self.poster_button = ui.Button(self.poster_rect, "Crear póster", primary=True)
        self.generate_button = ui.Button(self.generate_rect, "Generar", primary=True)
        self.slider = ui.Slider(self.slider_rect, 0, SEED_MAX, self.seed)
        self.static = ui._to_display_format(self.render_static())
        self.refresh()

    # ---------- geometría ----------
    def layout(self) -> None:
        W, H = self.size[0] / THEME.scale, self.size[1] / THEME.scale
        m = THEME.margin
        link = lambda t: ui.font(THEME.size_body, weight="medium").size(t)[0] / THEME.scale + 12
        self.back_rect = R(W - m - link("← Menú"), 14, link("← Menú"), 30)
        self.poster_rect = R(W - m - link("← Menú") - 26 - link("Crear póster"), 14, link("Crear póster"), 30)
        # Panel lateral (controles + evidencia) y zona principal (grilla o interpolación).
        self.panel = R(W - m - 250, 150, 250, H - 150 - 24)
        self.main = R(m, 166, W - 2 * m - 250 - 48, 392)
        self.cell = 104
        gap_x = (self.main.width / THEME.scale - GRID_COLS * self.cell) / (GRID_COLS - 1)
        self.cells = [R(m + c * (self.cell + gap_x), 166 + r * (self.cell + 30), self.cell, self.cell)
                      for r in range(GRID_ROWS) for c in range(GRID_COLS)]
        px0 = W - m - 250
        self.generate_rect = R(px0 + 250 - link("Generar"), 242, link("Generar"), 30)
        self.slider_rect = pygame.Rect(px(px0), px(300) - px(4), px(250), px(8))
        self.class_rects = [R(px0 + (i // 5) * 128, 364 + (i % 5) * 24, 120, 22) for i in range(len(CLASSES))]
        # Interpolación: tira de 10 pasos en una fila y, debajo, la animación con su explicación al lado.
        step = (self.main.width / THEME.scale - (INTERP_STEPS - 1) * 8) / INTERP_STEPS
        self.interp_strip = [R(m + i * (step + 8), 192, step, step) for i in range(INTERP_STEPS)]
        self.interp_big = R(m, 300, 220, 220)

    # ---------- generación (solo inferencia) ----------
    @property
    def G(self):
        return self.res[self.model]

    @property
    def labels(self):
        return self.class_idx if self.model == "cgan" else None

    def refresh(self) -> None:
        """Regenera las imágenes del estado actual (modelo, semilla, prenda, vista). Unos milisegundos en CPU."""
        from task_d_gan.models import generate, generate_from_seed, interpolate, noise
        self.images = generate_from_seed(self.G, self.seed, N_IMAGES, self.labels)
        self.predicted = None
        if self.res["clf"] is not None:
            with torch.no_grad():
                self.predicted = self.res["clf"](self.images).argmax(1).tolist()
        size = self.cells[0].width
        self.tiles = [self.tile(img, size) for img in self.images]
        if self.view == "interp":
            z0, z1 = noise(self.seed, 1, self.G.nz)[0], noise(self.seed + 1, 1, self.G.nz)[0]
            strip = generate(self.G, interpolate(z0, z1, INTERP_STEPS), self.labels)
            self.strip_tiles = [self.tile(img, self.interp_strip[0].width) for img in strip]
            # Los cuadros grandes se amplían la primera vez que se muestran (ampliar 48 de golpe tardaba ~0,5 s).
            self.frames = generate(self.G, interpolate(z0, z1, INTERP_FRAMES), self.labels)
            self.frame_tiles = {}
            self.anim_start = pygame.time.get_ticks()

    @staticmethod
    def tile(image: torch.Tensor, size: int) -> pygame.Surface:
        """Misma conversión que el póster: [-1, 1] -> 0..255, ampliación Lanczos y tinte marfil sobre oscuro."""
        from task_d_gan.poster import garment_surface
        return ui._to_display_format(garment_surface(image, size))

    def set_state(self, **changes) -> None:
        changed = any(getattr(self, k) != v for k, v in changes.items())
        for k, v in changes.items():
            setattr(self, k, v)
        if changed:
            self.refresh()

    def set_seed(self, seed: int) -> None:
        seed = int(max(0, min(SEED_MAX, seed)))
        if not self.slider.dragging:
            self.slider.value = seed
        self.set_state(seed=seed)

    # ---------- eventos ----------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        if self.poster is not None:   # con el póster abierto: cualquier clic o ESC lo cierra
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or \
                    (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
                self.poster = None
            return
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or self.back_button.handle_event(event):
            self.done = True
            return
        if self.res is None:
            return
        if self.generate_button.handle_event(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_g):
            self.generate_new()
        elif self.poster_button.handle_event(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_p):
            self.create_poster()
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
            self.set_seed(self.seed + (1 if event.key == pygame.K_RIGHT else -1))
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_UP, pygame.K_DOWN) and self.model == "cgan":
            self.set_state(class_idx=(self.class_idx + (1 if event.key == pygame.K_DOWN else -1)) % len(CLASSES))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.click_targets(event.pos):
            pass
        else:
            value = self.slider.handle_event(event)
            if self.slider.dragging or event.type == pygame.MOUSEBUTTONUP:
                self.set_seed(round(value))

    def click_targets(self, pos) -> bool:
        for key, rect in self.hit.items():
            if rect.collidepoint(pos):
                group, value = key
                if group == "class" and self.model != "cgan":
                    return False
                self.set_state(**{{"model": "model", "view": "view", "class": "class_idx"}[group]: value})
                return True
        return False

    def generate_new(self) -> None:
        """Botón Generar: una semilla nueva (la muestra el slider). Volver a esa semilla da las mismas imágenes."""
        seed = self.seed
        while seed == self.seed:
            seed = self.rng.randint(0, SEED_MAX)
        self.set_seed(seed)
        self.reveal_start = pygame.time.get_ticks()

    def create_poster(self) -> None:
        from task_d_gan.poster import build_poster, save_poster
        poster = build_poster(self.seed, self.res["cgan"], self.res["clf"])
        path = save_poster(poster, self.seed)
        h = self.size[1] - px(110)                       # deja libre la franja del aviso inferior
        w = int(poster.get_width() * h / poster.get_height())
        self.poster = (ui._to_display_format(pygame.transform.smoothscale(poster, (w, h))), path)
        try:
            shown = path.relative_to(config.ROOT).as_posix()
        except ValueError:
            shown = str(path)
        self.toast.show(f"Póster guardado en {shown}")

    # ---------- dibujo ----------
    def render_static(self) -> pygame.Surface:
        layer = ui.background(self.size).copy()
        m = px(THEME.margin)
        ui.blit_eyebrow(layer, "Tarea D · Imágenes promocionales con GAN", (m, px(22)), THEME.accent_text)
        ui.vhairline(layer, self.panel.left - px(24), self.panel.top, self.panel.bottom)
        return layer

    def draw(self, surface: pygame.Surface) -> None:
        if self.res is None:
            self.draw_error(surface)
            return
        surface.blit(self.static, (0, 0))
        self.hit = {}
        self.draw_header(surface)
        self.draw_view_tabs(surface)
        if self.view == "grid":
            self.draw_grid(surface)
        else:
            self.draw_interpolation(surface)
        self.draw_panel(surface)
        self.slider.draw(surface)
        self.poster_button.draw(surface)
        self.back_button.draw(surface)
        if self.poster is not None:
            self.draw_poster(surface)
        self.toast.draw(surface)
        self.update_cursor()
        self.fade.draw(surface)

    def draw_error(self, surface) -> None:
        surface.blit(ui.background(self.size), (0, 0))
        m = px(THEME.margin)
        ui.blit_eyebrow(surface, "Tarea D · Imágenes promocionales con GAN", (m, px(22)), THEME.accent_text)
        ui.blit_text(surface, "No se pudieron cargar los modelos entrenados", (m, px(50)), THEME.size_headline,
                     weight="serif")
        ui.blit_text(surface, self.error or "", (m, px(104)), THEME.size_label, THEME.text_muted)
        ui.blit_text(surface, "Ejecuta:  python -m task_d_gan.train", (m, px(132)), THEME.size_body)
        self.back_button.draw(surface)
        self.fade.draw(surface)

    def draw_header(self, surface) -> None:
        m = px(THEME.margin)
        text = headline(self.model, self.class_idx, N_IMAGES, self.view, self.seed)
        size = ui.fit_size(text, THEME.size_headline, THEME.size_title, self.size[0] - 2 * m)
        ui.blit_text(surface, text, (m, px(86)), size, weight="serif", anchor="bottomleft")
        ck = self.res[f"{self.model}_ckpt"]
        if self.model == "dcgan":
            what = "ruido z (64 valores) → generador → imagen 28×28 · la DCGAN elige la prenda sola"
        else:
            what = f"ruido z + clase «{CLASSES[self.class_idx]}» → generador → imagen 28×28"
        x = ui.blit_text(surface, f"Semilla {self.seed}", (m, px(100)), THEME.size_label, THEME.text,
                         weight="semibold").right
        ui.blit_text(surface, f"  ·  {what}  ·  {ck['epochs']} épocas en Fashion-MNIST", (x, px(100)),
                     THEME.size_label, THEME.text_muted)

    def word_selector(self, surface, group: str, options: dict, current, pos, size=THEME.size_body) -> int:
        """Palabras clicables; la activa en blanco con subrayado de acento (como el selector de la Tarea B)."""
        x, y = pos
        for value, label in options.items():
            active = value == current
            r = ui.blit_text(surface, label, (x, y), size, THEME.text if active else THEME.text_faint,
                             weight="semibold" if active else "medium")
            hit = r.inflate(px(12), px(10))
            if active:
                pygame.draw.rect(surface, THEME.accent, (r.left, r.bottom + px(3), r.width, max(2, px(2))))
            elif hit.collidepoint(self.mouse):
                ui.hairline(surface, r.left, r.right, r.bottom + px(3), (*THEME.text, 110))
            self.hit[(group, value)] = hit
            x = r.right + px(22)
        return x

    def draw_view_tabs(self, surface) -> None:
        self.word_selector(surface, "view", VIEWS, self.view, (px(THEME.margin), px(128)))

    def draw_grid(self, surface) -> None:
        reveal = (pygame.time.get_ticks() - self.reveal_start) / 450
        for i, (rect, tile) in enumerate(zip(self.cells, self.tiles)):
            t = min(1.0, max(0.0, reveal * 3 - i / N_IMAGES * 2))   # aparición escalonada tras "Generar"
            if t < 1:
                tile = tile.copy()
                tile.set_alpha(int(255 * t))
            surface.blit(tile, rect)
            if self.predicted is not None:
                pred = self.predicted[i]
                ok = self.model == "dcgan" or pred == self.class_idx
                ui.blit_text(surface, CLASSES[pred], (rect.left, rect.bottom + px(5)), THEME.size_caption,
                             THEME.text_muted if ok else THEME.warning)
        y = self.cells[-1].bottom + px(32)
        note = ("Debajo de cada imagen: la prenda que reconoce la CNN auxiliar (91 % en test)."
                if self.model == "dcgan" else
                "Debajo: la prenda que reconoce la CNN auxiliar; en dorado, si no coincide con la pedida.")
        ui.blit_text(surface, note, (px(THEME.margin), y), THEME.size_caption, THEME.text_faint)

    def draw_interpolation(self, surface) -> None:
        m = px(THEME.margin)
        ui.blit_eyebrow(surface, f"{INTERP_STEPS} pasos de z, de la semilla {self.seed} a la {self.seed + 1}",
                        (m, px(166)))
        t_cycle = ((pygame.time.get_ticks() - self.anim_start) % INTERP_MS) / INTERP_MS
        t = 1 - abs(1 - 2 * t_cycle)                     # ida y vuelta: 0 -> 1 -> 0
        t = t * t * (3 - 2 * t)                          # suavizado en los extremos
        active = int(round(t * (INTERP_STEPS - 1)))
        for i, (rect, tile) in enumerate(zip(self.interp_strip, self.strip_tiles)):
            surface.blit(tile, rect)
            if i == active:
                pygame.draw.rect(surface, THEME.accent, (rect.left, rect.bottom + px(5), rect.width, max(2, px(2))))
        first, last = self.interp_strip[0], self.interp_strip[-1]
        ui.blit_text(surface, f"semilla {self.seed}", (first.left, first.bottom + px(12)), THEME.size_caption,
                     THEME.text_faint)
        ui.blit_text(surface, f"semilla {self.seed + 1}", (last.right, last.bottom + px(12)), THEME.size_caption,
                     THEME.text_faint, anchor="topright")
        # Animación: el mismo camino con 48 cuadros (cada uno se amplía solo la primera vez que se muestra).
        frame = int(round(t * (INTERP_FRAMES - 1)))
        if frame not in self.frame_tiles:
            self.frame_tiles[frame] = self.tile(self.frames[frame], self.interp_big.width)
        big = self.interp_big
        surface.blit(self.frame_tiles[frame], big)
        x = big.right + px(28)
        width = self.main.right - x
        r = ui.blit_text(surface, f"t = {t:.2f}".replace(".", ","), (x, big.top), THEME.size_number, weight="serif")
        y = r.bottom + px(4)
        ui.blit_text(surface, "z(t) = (1 − t) · z_inicio + t · z_fin", (x, y), THEME.size_label, THEME.text)
        y += px(34)
        text = ("Interpolación lineal entre dos vectores de ruido. Si las prendas intermedias son plausibles "
                "y cambian poco a poco, el generador aprendió un espacio continuo: no memorizó imágenes sueltas.")
        if self.model == "cgan":
            text += f" La clase queda fija («{CLASSES[self.class_idx]}»): solo cambia el estilo."
        else:
            text += " En la DCGAN el camino puede pasar de una prenda a otra: z no separa las clases."
        for line in ui.wrap_text(text, THEME.size_label, width):
            ui.blit_text(surface, line, (x, y), THEME.size_label, THEME.text_muted)
            y += px(21)

    def draw_panel(self, surface) -> None:
        p = self.panel
        x = p.left
        ui.blit_eyebrow(surface, "Modelo", (x, p.top))
        self.word_selector(surface, "model", MODELS, self.model, (x, p.top + px(22)))
        # Semilla del ruido: número grande + Generar + slider.
        ui.blit_eyebrow(surface, "Semilla del ruido", (x, px(214)))
        ui.blit_text(surface, str(self.seed), (x, self.generate_rect.centery), THEME.size_number + 4,
                     weight="serif", anchor="midleft")
        self.generate_button.draw(surface)
        ui.blit_text(surface, "misma semilla = mismas imágenes", (x, px(314)), THEME.size_caption,
                     THEME.text_faint)
        # Prenda (solo cGAN).
        ui.blit_eyebrow(surface, "Prenda (cGAN)", (x, px(342)))
        if self.model == "cgan":
            for i, rect in enumerate(self.class_rects):
                active = i == self.class_idx
                hovered = rect.collidepoint(self.mouse)
                color = THEME.accent_text if active else (THEME.text if hovered else THEME.text_muted)
                r = ui.blit_text(surface, CLASSES[i], (rect.left, rect.top), THEME.size_label, color,
                                 weight="semibold" if active else "regular")
                if active:
                    pygame.draw.rect(surface, THEME.accent, (r.left, r.bottom + px(1), r.width, max(2, px(2))))
                self.hit[("class", i)] = rect
        else:
            y = px(366)
            for line in ("La DCGAN no recibe la clase:", "elige la prenda sola. Cambia a",
                         "cGAN para pedir una prenda."):
                ui.blit_text(surface, line, (x, y), THEME.size_label, THEME.text_faint)
                y += px(21)
        ui.hairline(surface, x, p.right, px(492))
        self.draw_evidence(surface, x, px(504))

    def draw_evidence(self, surface, x: int, y: int) -> None:
        s = self.res["summary"]
        ui.blit_eyebrow(surface, "Evidencia", (x, y))
        if s is None:
            ui.blit_text(surface, "Ejecuta python -m task_d_gan.evaluate", (x, y + px(24)), THEME.size_label,
                         THEME.text_faint)
            return
        if self.model == "dcgan":
            d = s["dcgan"]
            big = ui.blit_text(surface, f"{d['classes_covered']}/10", (x, y + px(58)), THEME.size_number,
                               weight="serif", anchor="bottomleft")
            ui.blit_text(surface, "clases cubiertas", (big.right + px(8), big.bottom - px(8)), THEME.size_label,
                         THEME.text_muted, anchor="bottomleft")
            lines = [f"cada una, entre el {fmt_pct(d['min_class_share_pct'])} y el "
                     f"{fmt_pct(d['max_class_share_pct'])}", "de 10 000 muestras: sin colapso"]
        else:
            c = s["cgan"]
            big = ui.blit_text(surface, fmt_pct(c["control_accuracy"] * 100), (x, y + px(58)), THEME.size_number,
                               weight="serif", anchor="bottomleft")
            ui.blit_text(surface, "son de la clase pedida", (big.right + px(8), big.bottom - px(8)),
                         THEME.size_label, THEME.text_muted, anchor="bottomleft")
            name = CLASSES[self.class_idx]
            lines = [f"{name}: {fmt_pct(c['control_by_class_pct'][name])} de 1000 pedidas",
                     f"referencia, CNN en reales: {fmt_pct(s['classifier']['per_class'][name])}"]
        yy = y + px(64)
        for line in lines:
            ui.blit_text(surface, line, (x, yy), THEME.size_label, THEME.text_muted)
            yy += px(20)

    def draw_poster(self, surface) -> None:
        veil = pygame.Surface(self.size, pygame.SRCALPHA)
        veil.fill((*THEME.bg, 245))
        surface.blit(veil, (0, 0))
        img, path = self.poster
        rect = img.get_rect(midtop=(self.size[0] // 2, px(18)))
        surface.blit(img, rect)
        ui.blit_text(surface, "Clic o ESC para cerrar", (rect.right + px(20), rect.bottom), THEME.size_label,
                     THEME.text_faint, anchor="bottomleft")

    def update_cursor(self) -> None:
        clickable = [*self.hit.values(), self.poster_button.rect, self.back_button.rect, self.generate_button.rect]
        hand = self.poster is None and (any(r.collidepoint(self.mouse) for r in clickable) or self.slider.dragging)
        if hand != self._cursor:
            self._cursor = hand
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hand else pygame.SYSTEM_CURSOR_ARROW)
            except pygame.error:  # sin ventana real (pruebas headless)
                pass


def run(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    try:
        view = GanScreen(load_cached(), size=screen.get_size())
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        view = GanScreen(None, str(exc), size=screen.get_size())
    while not view.done:
        clock.tick(config.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.event.post(pygame.event.Event(pygame.QUIT))  # el menú también debe cerrarse
                return
            view.handle_event(event)
        view.draw(screen)
        pygame.display.flip()
