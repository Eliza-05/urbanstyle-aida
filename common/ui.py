"""Tema visual, escala DPI y componentes Pygame compartidos por todas las pantallas (responsable: Elizabeth).

Reglas para cualquier pantalla (A, B, C, D y menú):
  * Diseña en píxeles LÓGICOS (ventana base de 1024×640) y convierte con `ui.px(valor)`: así la app se ve
    nítida en Windows con escala 150 % (la ventana se crea en píxeles físicos con `ui.create_window`).
  * Toma colores y tamaños SIEMPRE de `THEME`; los tamaños de letra se pasan en px lógicos.
  * Texto: `ui.blit_text` / `ui.render_text` (Inter con antialias, cacheado). Tarjetas: `ui.draw_card`.
    Líneas y círculos con antialias: `ui.aa_polyline`, `ui.aa_line`, `ui.aa_circle`.
  * Lo costoso (fondo, sombras, degradados, tarjetas) se pre-renderiza y se cachea automáticamente.

Ejemplo:
    screen = ui.create_window((1024, 640), "UrbanStyle")        # en main.py, una sola vez
    ui.draw_card(screen, pygame.Rect(ui.px(16), ui.px(16), ui.px(300), ui.px(120)))
    ui.blit_text(screen, "Hola", (ui.px(32), ui.px(32)), THEME.size_heading, weight="semibold")
    boton = ui.Button(rect, "Entrenar", primary=True)              # misma API de siempre

Fuente: Inter (SIL Open Font License 1.1), en assets/fonts/ junto con OFL.txt.
"""
import os
import sys

import numpy as np
import pygame
import pygame.gfxdraw

from common import config

FONT_DIR = config.ROOT / "assets" / "fonts"
FONT_FILES = {"regular": "Inter-Regular.ttf", "medium": "Inter-Medium.ttf",
              "semibold": "Inter-SemiBold.ttf", "bold": "Inter-Bold.ttf"}


def _hex(code: str) -> tuple:
    return tuple(int(code[i:i + 2], 16) for i in (1, 3, 5))


class THEME:
    """Tema único de UrbanStyle (modo oscuro, estilo dashboard). Se usa como `ui.THEME.accent`."""

    scale = 1.0  # factor DPI (1.5 en Windows al 150 %); lo fija create_window/set_scale

    # Fondo y superficies (las superficies son RGBA translúcidas: efecto "vidrio" sobre el degradado)
    bg = _hex("#0B0F19")
    glow = _hex("#7C5CFF")               # resplandor violeta en la esquina superior derecha
    glow_2 = _hex("#D946EF")             # resplandor fucsia, muy tenue, abajo a la izquierda
    surface = (255, 255, 255, 13)
    surface_hover = (255, 255, 255, 22)
    surface_alt = (255, 255, 255, 9)     # zonas dentro de una tarjeta
    border = (255, 255, 255, 28)
    grid = (255, 255, 255, 16)
    shadow = (2, 4, 12)

    # Texto (contraste ≥ 5:1 sobre el fondo incluso en el gris más tenue)
    text = _hex("#E6E8F0")
    text_muted = _hex("#A3ACC2")
    text_faint = _hex("#8790A8")

    # Marca y estados
    accent = _hex("#7C5CFF")             # violeta UrbanStyle: botón principal, slider, activos
    accent_2 = _hex("#D946EF")           # fucsia: final de los degradados de marca
    accent_text = _hex("#C4B5FD")        # texto sobre píldoras violetas
    success = _hex("#34D399")
    warning = _hex("#F59E0B")
    danger = _hex("#F87171")

    # Colores por modelo (iguales en gráfica, leyenda, tooltip y tabla)
    series_colors = {
        "actual": _hex("#64748B"),
        "lstm": _hex("#22D3EE"),
        "moving_average": _hex("#F59E0B"),
        "hist_mean": _hex("#34D399"),
        "naive": _hex("#94A3B8"),
    }

    # Tipografía en px LÓGICOS (mínimo 14 para que se lea al compartir pantalla)
    size_hero = 54
    size_kpi = 30
    size_title = 26
    size_heading = 17
    size_body = 15
    size_label = 14
    size_caption = 14

    # Espaciado y forma en px lógicos
    pad_xs, pad_sm, pad_md, pad_lg, pad_xl = 4, 8, 12, 16, 24
    radius = 16
    radius_sm = 10


# ---------------------------------------------------------------------------
# Escala DPI y ventana
# ---------------------------------------------------------------------------

def px(value: float) -> int:
    """Píxeles lógicos -> físicos según la escala DPI."""
    return int(round(value * THEME.scale))


def set_scale(scale: float) -> None:
    THEME.scale = scale
    _clear_caches()


def _enable_dpi_awareness() -> None:
    """Windows: que el sistema NO estire la ventana (eso la vuelve borrosa). En Mac/Linux no hace nada."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _system_scale() -> float:
    if sys.platform != "win32":
        return 1.0
    import ctypes
    try:
        return ctypes.windll.user32.GetDpiForSystem() / 96
    except (AttributeError, OSError):
        return 1.0


def _work_area():
    """Área útil de la pantalla en píxeles físicos (sin barra de tareas)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
            return rect.right - rect.left, rect.bottom - rect.top
    sizes = pygame.display.get_desktop_sizes()
    return sizes[0] if sizes else (10_000, 10_000)


def create_window(logical_size, caption: str = "UrbanStyle") -> pygame.Surface:
    """Crea la ventana en píxeles físicos (logical_size × escala DPI) y fija THEME.scale.

    Si la ventana no cabe en el área útil de la pantalla, reduce la escala. La variable de entorno
    URBANSTYLE_SCALE fuerza una escala (útil para capturas o para probar otra escala).
    """
    _enable_dpi_awareness()
    pygame.init()
    forced = os.environ.get("URBANSTYLE_SCALE")
    if forced:
        scale = float(forced)
    elif pygame.display.get_driver() == "dummy":
        scale = 1.0
    else:
        scale = _system_scale()
        work_w, work_h = _work_area()
        title_bar = 40 * scale
        w, h = logical_size
        while scale > 1.0 and (w * scale > work_w - 16 or h * scale + title_bar > work_h - 8):
            scale = round(scale - 0.05, 2)
    set_scale(max(1.0, scale))
    surface = pygame.display.set_mode((px(logical_size[0]), px(logical_size[1])))
    pygame.display.set_caption(caption)
    return surface


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

_font_cache: dict = {}
_text_cache: dict = {}
_wrap_cache: dict = {}
_surface_cache: dict = {}
_TEXT_CACHE_MAX = 4000


def _clear_caches() -> None:
    """Fuentes y superficies dependen de la escala y dejan de ser válidas tras pygame.quit()."""
    for cache in (_font_cache, _text_cache, _wrap_cache, _surface_cache):
        cache.clear()


def font(size: int, bold: bool = False, weight: str | None = None) -> pygame.font.Font:
    """Inter del tamaño (px lógicos) y peso pedidos: regular, medium, semibold o bold."""
    weight = weight or ("bold" if bold else "regular")
    key = (px(size), weight)
    if key not in _font_cache:
        if not _font_cache:  # primera fuente de esta sesión de pygame
            pygame.register_quit(_clear_caches)
        path = FONT_DIR / FONT_FILES[weight]
        _font_cache[key] = pygame.font.Font(str(path) if path.exists() else None, px(size))
    return _font_cache[key]


def render_text(text: str, size: int, color=THEME.text, bold: bool = False, weight: str | None = None):
    """Texto con antialias, cacheado (renderizar texto es lo más caro de cada cuadro)."""
    key = (text, size, color, bold, weight, THEME.scale)
    surf = _text_cache.get(key)
    if surf is None:
        if len(_text_cache) > _TEXT_CACHE_MAX:
            _text_cache.clear()
        surf = _to_display_format(font(size, bold, weight).render(text, True, color))
        _text_cache[key] = surf
    return surf


def blit_text(surface, text, pos, size=THEME.size_body, color=THEME.text, bold=False, anchor="topleft",
              weight=None):
    """Dibuja texto anclado en `pos` (topleft, midtop, center, midright, ...) y devuelve su Rect."""
    img = render_text(text, size, color, bold, weight)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def wrap_text(text: str, size: int, width: int, bold: bool = False, weight: str | None = None) -> list[str]:
    """Parte un texto en líneas que caben en `width` píxeles físicos (cacheado)."""
    key = (text, size, width, bold, weight, THEME.scale)
    if key not in _wrap_cache:
        f = font(size, bold, weight)
        lines, line = [], ""
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if f.size(candidate)[0] <= width or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        _wrap_cache[key] = lines
    return _wrap_cache[key]


# ---------------------------------------------------------------------------
# Superficies pre-renderizadas (cacheadas por tamaño)
# ---------------------------------------------------------------------------

def _to_display_format(surf: pygame.Surface) -> pygame.Surface:
    """Convierte al formato de la ventana (si ya existe): los blits son varias veces más rápidos."""
    if pygame.display.get_surface() is None:
        return surf
    return surf.convert_alpha() if surf.get_flags() & pygame.SRCALPHA else surf.convert()


def _cached(key, build):
    if key not in _surface_cache:
        _surface_cache[key] = _to_display_format(build())
    return _surface_cache[key]


def rounded_surface(size, radius: int, fill, border=None, border_width: int = 1) -> pygame.Surface:
    """Rectángulo redondeado con bordes suavizados (se dibuja a 3× y se reduce). Colores RGBA."""
    w, h = int(size[0]), int(size[1])

    def build():
        ss = 3
        big = pygame.Surface((w * ss, h * ss), pygame.SRCALPHA)
        big.fill((*fill[:3], 0))  # mismo RGB con alfa 0: evita un borde oscuro al suavizar
        pygame.draw.rect(big, fill, big.get_rect(), border_radius=radius * ss)
        if border:
            pygame.draw.rect(big, border, big.get_rect(), border_width * ss, border_radius=radius * ss)
        return pygame.transform.smoothscale(big, (w, h))

    return _cached(("round", w, h, radius, tuple(fill), tuple(border) if border else None, border_width), build)


def _blur(surface: pygame.Surface, radius: int) -> pygame.Surface:
    """Desenfoque barato: reducir y volver a ampliar con suavizado (dos pasadas)."""
    w, h = surface.get_size()
    factor = max(2, radius // 2)
    small = (max(1, w // factor), max(1, h // factor))
    for _ in range(2):
        surface = pygame.transform.smoothscale(pygame.transform.smoothscale(surface, small), (w, h))
    return surface


def shadow_surface(size, radius: int, blur: int, alpha: int) -> pygame.Surface:
    """Sombra difusa de una tarjeta; mide (w + 2·blur, h + 2·blur)."""
    w, h = int(size[0]), int(size[1])

    def build():
        layer = pygame.Surface((w + 2 * blur, h + 2 * blur), pygame.SRCALPHA)
        layer.fill((*THEME.shadow, 0))
        pygame.draw.rect(layer, (*THEME.shadow, alpha), (blur, blur, w, h), border_radius=radius)
        return _blur(layer, blur)

    return _cached(("shadow", w, h, radius, blur, alpha), build)


def gradient_surface(size, color_a, color_b, radius: int = 0, vertical: bool = False, alpha=(255, 255)):
    """Degradado lineal (horizontal por defecto) de color_a a color_b, opcionalmente redondeado."""
    w, h = int(size[0]), int(size[1])

    def build():
        n = h if vertical else w
        t = np.linspace(0.0, 1.0, max(n, 1))[:, None]
        rgb = (np.array(color_a[:3]) * (1 - t) + np.array(color_b[:3]) * t).astype(np.uint8)
        a = (alpha[0] * (1 - t[:, 0]) + alpha[1] * t[:, 0]).astype(np.uint8)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        colors = pygame.surfarray.pixels3d(surf)
        alphas = pygame.surfarray.pixels_alpha(surf)
        if vertical:
            colors[:] = rgb[None, :, :]
            alphas[:] = a[None, :]
        else:
            colors[:] = rgb[:, None, :]
            alphas[:] = a[:, None]
        del colors, alphas
        if radius:
            mask = rounded_surface((w, h), radius, (255, 255, 255, 255))
            surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return surf

    return _cached(("grad", w, h, tuple(color_a), tuple(color_b), radius, vertical, tuple(alpha)), build)


def background(size) -> pygame.Surface:
    """Fondo casi negro con resplandor radial violeta (arriba a la derecha) y fucsia muy tenue (abajo a la izquierda)."""
    w, h = int(size[0]), int(size[1])

    def build():
        x = np.arange(w)[:, None] / w
        y = np.arange(h)[None, :] / h
        aspect = w / h
        img = np.zeros((w, h, 3)) + np.array(THEME.bg)
        for (cx, cy), color, strength, reach in (((0.95, -0.15), THEME.glow, 0.26, 0.95),
                                                 ((0.0, 1.1), THEME.glow_2, 0.07, 0.75)):
            d = np.sqrt(((x - cx) * aspect) ** 2 + (y - cy) ** 2) / (reach * aspect)
            weight = np.clip(1 - d, 0, 1) ** 2 * strength
            img = img * (1 - weight[..., None]) + np.array(color) * weight[..., None]
        img += np.random.default_rng(0).uniform(-0.6, 0.6, img.shape)  # tramado: evita bandas visibles
        return pygame.surfarray.make_surface(np.clip(img, 0, 255).astype(np.uint8))

    return _cached(("bg", w, h), build)


def draw_card(surface, rect, radius=None, fill=THEME.surface, border=THEME.border, shadow: bool = True,
              elevation: float = 0.0) -> None:
    """Tarjeta tipo vidrio: sombra difusa (más grande con `elevation` 0..1), relleno translúcido y borde de 1 px."""
    rect = pygame.Rect(rect)
    radius = px(THEME.radius) if radius is None else radius
    if shadow:
        blur = px(14)
        base = shadow_surface(rect.size, radius, blur, 110)
        surface.blit(base, (rect.left - blur, rect.top - blur + px(4)))
        if elevation > 0:
            big_blur = px(22)
            lifted = shadow_surface(rect.size, radius, big_blur, 150)
            lifted.set_alpha(int(255 * elevation))
            surface.blit(lifted, (rect.left - big_blur, rect.top - big_blur + px(10)))
            lifted.set_alpha(255)
    surface.blit(rounded_surface(rect.size, radius, fill, border, max(1, px(1))), rect)


def pill(surface, text, pos, color=THEME.accent, text_color=None, size=THEME.size_caption, anchor="topleft",
         weight="semibold"):
    """Etiqueta en forma de píldora con fondo translúcido del color dado. Devuelve su Rect."""
    img = render_text(text, size, text_color or THEME.accent_text, weight=weight)
    rect = img.get_rect().inflate(px(20), px(8))
    setattr(rect, anchor, pos)
    surface.blit(rounded_surface(rect.size, rect.height // 2, (*color[:3], 46), (*color[:3], 110)), rect)
    surface.blit(img, img.get_rect(center=rect.center))
    return rect


# ---------------------------------------------------------------------------
# Dibujo con antialias
# ---------------------------------------------------------------------------

def aa_circle(surface, color, center, radius) -> None:
    x, y, r = int(round(center[0])), int(round(center[1])), int(round(radius))
    pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    pygame.gfxdraw.aacircle(surface, x, y, r, color)


def aa_line(surface, color, p1, p2, width: int = 1) -> None:
    """Línea suavizada de cualquier grosor (cuadrilátero con bordes antialias + extremos redondos)."""
    if width <= 1:
        pygame.draw.aaline(surface, color, p1, p2)
        return
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ox, oy = -dy / length * width / 2, dx / length * width / 2
    quad = [(x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (x2 - ox, y2 - oy), (x1 - ox, y1 - oy)]
    pygame.gfxdraw.aapolygon(surface, quad, color)
    pygame.gfxdraw.filled_polygon(surface, quad, color)


def aa_polyline(surface, color, points, width: int = 1) -> None:
    """Polilínea suavizada con uniones redondeadas."""
    for a, b in zip(points, points[1:]):
        aa_line(surface, color, a, b, width)
    if width > 2:
        for p in points:
            aa_circle(surface, color, p, width / 2 - 0.5)


def dashed_polyline(surface, color, points, width: int, dash: int, gap: int | None = None) -> None:
    """Línea discontinua suavizada a lo largo de una polilínea (pygame no la trae)."""
    gap = dash if gap is None else gap
    draw, left = True, dash
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        pos = 0.0
        while pos < length:
            step = min(left, length - pos)
            if draw and step > 0.5:
                a = (x1 + (x2 - x1) * pos / length, y1 + (y2 - y1) * pos / length)
                b = (x1 + (x2 - x1) * (pos + step) / length, y1 + (y2 - y1) * (pos + step) / length)
                aa_line(surface, color, a, b, width)
            pos += step
            left -= step
            if left <= 0:
                draw = not draw
                left = dash if draw else gap


def glow_surface(radius: int, color, alpha: int = 90) -> pygame.Surface:
    """Halo radial suave (para el control del slider o elementos activos)."""
    def build():
        size = radius * 2
        y, x = np.ogrid[:size, :size]
        d = np.sqrt((x - radius + 0.5) ** 2 + (y - radius + 0.5) ** 2) / radius
        a = (np.clip(1 - d, 0, 1) ** 2 * alpha).astype(np.uint8)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((*color[:3], 0))
        pygame.surfarray.pixels_alpha(surf)[:] = a.T
        return surf

    return _cached(("glow", radius, tuple(color), alpha), build)


# ---------------------------------------------------------------------------
# Componentes
# ---------------------------------------------------------------------------

class _Hover:
    """Progreso de hover animado (0 → 1 en `duration` s) para microinteracciones."""

    def __init__(self, duration: float = 0.18):
        self.duration = duration
        self.value = 0.0
        self.last = pygame.time.get_ticks()

    def step(self, target: bool) -> float:
        now = pygame.time.get_ticks()
        delta = (now - self.last) / 1000 / self.duration
        self.last = now
        self.value = min(1.0, self.value + delta) if target else max(0.0, self.value - delta)
        return self.value


class Button:
    """Botón con hover animado. API: Button(rect, label, font=None), draw(surface), handle_event(event) -> bool.

    Opcionales: `subtitle` (tarjeta con segunda línea), `badge` (letra en un círculo de acento) y
    `primary` (degradado violeta→fucsia de la marca). `rect` va en píxeles físicos (usa ui.px).
    """

    def __init__(self, rect, label: str, font: pygame.font.Font | None = None, subtitle: str | None = None,
                 badge: str | None = None, primary: bool = False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font
        self.subtitle = subtitle
        self.badge = badge
        self.primary = primary
        self.draw_rect = self.rect
        self._hover = _Hover()

    def hovered(self) -> bool:
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def draw(self, surface: pygame.Surface) -> None:
        h = self._hover.step(self.hovered())
        lift = int(round(px(2) * h))
        rect = self.draw_rect = self.rect.move(0, -lift)  # draw_rect: posición visible (con elevación)
        if self.primary:
            radius = px(THEME.radius_sm)
            surface.blit(shadow_surface(rect.size, radius, px(12), 140), (rect.left - px(12), rect.top - px(8)))
            surface.blit(gradient_surface(rect.size, THEME.accent, THEME.accent_2, radius), rect)
            if h > 0:
                shine = rounded_surface(rect.size, radius, (255, 255, 255, int(40 * h)))
                surface.blit(shine, rect)
            text_color = (255, 255, 255)
        else:
            card = bool(self.subtitle)
            fill = tuple(int(a + (b - a) * h) for a, b in zip(THEME.surface, THEME.surface_hover))
            border = (*THEME.accent, int(28 + 140 * h)) if h > 0 else THEME.border
            draw_card(surface, rect, px(THEME.radius if card else THEME.radius_sm), fill, border,
                      shadow=True, elevation=h)
            text_color = THEME.text

        x = rect.left + px(THEME.pad_lg)
        if self.badge:
            r = px(20)
            center = (x + r, rect.centery)
            if h > 0:
                halo = glow_surface(int(r * 1.9), THEME.accent, int(120 * h))
                surface.blit(halo, halo.get_rect(center=center))
            surface.blit(gradient_surface((2 * r, 2 * r), THEME.accent, THEME.accent_2, r), (x, rect.centery - r))
            blit_text(surface, self.badge, center, THEME.size_heading, (255, 255, 255), weight="bold", anchor="center")
            x += 2 * r + px(THEME.pad_md + 2)

        if self.subtitle:
            blit_text(surface, self.label, (x, rect.centery - px(1)), THEME.size_heading, text_color,
                      weight="semibold", anchor="bottomleft")
            blit_text(surface, self.subtitle, (x, rect.centery + px(4)), THEME.size_label, THEME.text_muted)
        elif self.font is not None:  # compatibilidad: fuente propia, texto centrado
            img = self.font.render(self.label, True, text_color)
            surface.blit(img, img.get_rect(center=rect.center))
        else:
            blit_text(surface, self.label, rect.center, THEME.size_body, text_color, weight="semibold",
                      anchor="center")

    def handle_event(self, event: pygame.event.Event) -> bool:
        """True si se hizo clic izquierdo sobre el botón."""
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


class Slider:
    """Slider horizontal. API: Slider(rect, min_value, max_value, value), handle_event(event) -> valor, draw(surface).

    Mientras se arrastra, el control sigue al mouse (`dragging` = True); quien lo usa puede redondear el
    valor al soltar para encajar en posiciones discretas. `rect` va en píxeles físicos.
    """

    def __init__(self, rect, min_value: float, max_value: float, value: float | None = None):
        self.rect = pygame.Rect(rect)
        self.min_value = min_value
        self.max_value = max_value
        self.value = min_value if value is None else value
        self.dragging = False
        self._hover = _Hover(0.15)

    def _set_from_x(self, x: int) -> None:
        t = (x - self.rect.left) / max(1, self.rect.width)
        t = min(1.0, max(0.0, t))
        self.value = self.min_value + t * (self.max_value - self.min_value)

    def x_of(self, value: float) -> int:
        """Posición x en pantalla de un valor (útil para dibujar marcas)."""
        span = (self.max_value - self.min_value) or 1
        return int(self.rect.left + (value - self.min_value) / span * self.rect.width)

    def _hit(self, pos) -> bool:
        return self.rect.inflate(px(28), px(32)).collidepoint(pos)

    def handle_event(self, event: pygame.event.Event) -> float:
        """Procesa clic/arrastre y devuelve el valor actual."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._hit(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
        return self.value

    def draw(self, surface: pygame.Surface) -> None:
        track_h = px(6)
        track = pygame.Rect(self.rect.left, self.rect.centery - track_h // 2, self.rect.width, track_h)
        surface.blit(rounded_surface(track.size, track_h // 2, (255, 255, 255, 26)), track)
        x = self.x_of(self.value)
        filled = gradient_surface(track.size, THEME.accent, THEME.accent_2, track_h // 2)
        surface.blit(filled, track.topleft, pygame.Rect(0, 0, max(0, x - track.left), track_h))
        h = self._hover.step(self.dragging or self._hit(pygame.mouse.get_pos()))
        if h > 0:
            halo = glow_surface(px(22), THEME.accent, int(150 * h))
            surface.blit(halo, halo.get_rect(center=(x, self.rect.centery)))
        aa_circle(surface, THEME.accent, (x, self.rect.centery), px(11))
        aa_circle(surface, (255, 255, 255), (x, self.rect.centery), px(7.5))


class Toast:
    """Aviso temporal (p. ej. "Archivo exportado"). show(texto), draw(surface) en cada cuadro."""

    def __init__(self, duration_ms: int = 3500):
        self.duration_ms = duration_ms
        self.message = ""
        self.until = 0

    def show(self, message: str) -> None:
        self.message = message
        self.until = pygame.time.get_ticks() + self.duration_ms

    def draw(self, surface: pygame.Surface) -> None:
        remaining = self.until - pygame.time.get_ticks()
        if not self.message or remaining <= 0:
            return
        img = render_text(self.message, THEME.size_body, THEME.text, weight="medium")
        box = img.get_rect().inflate(px(64), px(24))
        box.midbottom = (surface.get_width() // 2, surface.get_height() - px(28))
        layer = pygame.Surface((box.width + px(40), box.height + px(40)), pygame.SRCALPHA)
        inner = pygame.Rect(px(20), px(16), box.width, box.height)
        draw_card(layer, inner, px(14), (22, 27, 40, 245), (*THEME.success, 150))
        aa_circle(layer, THEME.success, (inner.left + px(22), inner.centery), px(9))
        blit_text(layer, "✓", (inner.left + px(22), inner.centery), THEME.size_label, THEME.bg, weight="bold",
                  anchor="center")
        layer.blit(img, img.get_rect(midleft=(inner.left + px(40), inner.centery)))
        layer.set_alpha(255 if remaining > 400 else int(255 * remaining / 400))  # se desvanece al final
        surface.blit(layer, (box.left - px(20), box.top - px(16)))


class FadeIn:
    """Aparición suave al entrar a una pantalla: un velo del color de fondo que se desvanece."""

    def __init__(self, duration_ms: int = 220):
        self.duration_ms = duration_ms
        self.start = pygame.time.get_ticks()
        self.veil = None

    def draw(self, surface: pygame.Surface) -> None:
        t = (pygame.time.get_ticks() - self.start) / self.duration_ms
        if t >= 1:
            return
        if self.veil is None or self.veil.get_size() != surface.get_size():
            self.veil = pygame.Surface(surface.get_size())
            self.veil.fill(THEME.bg)
        self.veil.set_alpha(int(255 * (1 - t) ** 2))
        surface.blit(self.veil, (0, 0))
