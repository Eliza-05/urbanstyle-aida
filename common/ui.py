"""Tema visual, escala DPI y componentes Pygame compartidos por todas las pantallas (responsable: Elizabeth).

Lenguaje visual de UrbanStyle: EDITORIAL (periodismo de datos + revista de moda), modo oscuro.
  * Sin tarjetas con borde ni sombra: las secciones se separan con espacio, jerarquía tipográfica y
    líneas finas (`ui.hairline`). La gráfica es la protagonista.
  * Dos familias: DM Serif Display (`weight="serif"`) para titulares y números grandes; Inter para lo demás.
  * Un solo acento de marca (`THEME.accent`); el resto en blancos rotos y grises cálidos.

Reglas para cualquier pantalla (A, B, C, D y menú):
  * Diseña en píxeles LÓGICOS (ventana base 1024×640) y convierte con `ui.px(valor)`: la ventana se crea
    en píxeles físicos con `ui.create_window`, así se ve nítida en Windows al 150 %.
  * Colores y tamaños SIEMPRE desde `THEME`; tamaños de letra en px lógicos (mínimo 14).
  * Texto: `ui.blit_text(..., weight=...)` (cacheado, con antialias); mayúsculas espaciadas: `ui.blit_eyebrow`.
  * Líneas y círculos con antialias: `ui.aa_polyline`, `ui.aa_line`, `ui.aa_circle`, `ui.dashed_polyline`.
  * Lo costoso (fondos, degradados, textos) se pre-renderiza y se cachea automáticamente.

Ejemplo:
    screen = ui.create_window((1024, 640), "UrbanStyle")        # en main.py, una sola vez
    ui.blit_eyebrow(screen, "TAREA A · RUTAS", (ui.px(32), ui.px(22)))
    ui.blit_text(screen, "Titular", (ui.px(32), ui.px(44)), THEME.size_headline, weight="serif")
    ui.hairline(screen, ui.px(32), ui.px(990), ui.px(100))
    enlace = ui.Button(rect, "Exportar CSV", primary=True)        # botón de texto (misma API de siempre)

Fuentes (SIL Open Font License 1.1, en assets/fonts/ con su licencia): Inter y DM Serif Display.
"""
import os
import sys

import numpy as np
import pygame
import pygame.gfxdraw

from common import config

FONT_DIR = config.ROOT / "assets" / "fonts"
FONT_FILES = {"regular": "Inter-Regular.ttf", "medium": "Inter-Medium.ttf",
              "semibold": "Inter-SemiBold.ttf", "bold": "Inter-Bold.ttf",
              "serif": "DMSerifDisplay-Regular.ttf", "serif_italic": "DMSerifDisplay-Italic.ttf"}


def _hex(code: str) -> tuple:
    return tuple(int(code[i:i + 2], 16) for i in (1, 3, 5))


class THEME:
    """Tema único de UrbanStyle (editorial, modo oscuro). Se usa como `ui.THEME.accent`."""

    scale = 1.0  # factor DPI × config.UI_ZOOM (1.25 en Mac; 1.875 en Windows al 150 % si cabe); create_window

    # Fondo y líneas
    bg = _hex("#0F0E0D")                 # casi negro cálido
    bg_raised = _hex("#1A1816")          # fondo sólido de tooltips y avisos
    chart_bg = (255, 248, 240, 6)        # único fondo sutil permitido: la zona de la gráfica
    hairline_color = (239, 235, 228, 38)  # líneas finas de separación (1 px, baja opacidad)
    grid = (239, 235, 228, 20)

    # Texto (contraste ≥ 5,5:1 sobre el fondo incluso en el gris más tenue)
    text = _hex("#EFEBE4")               # blanco roto
    text_muted = _hex("#B7B0A8")         # gris cálido
    text_faint = _hex("#8F8983")         # gris cálido tenue

    # Un solo acento de marca
    accent = _hex("#FF5A36")             # bermellón UrbanStyle: LSTM, selección, enlaces activos
    accent_text = _hex("#FF7A5C")        # el acento para texto pequeño (más legible)
    accent_2 = accent                    # compatibilidad: ya no se usan degradados de marca
    success = _hex("#7E9C89")
    warning = _hex("#B09A74")
    danger = _hex("#E5484D")

    # Colores por modelo: la LSTM con el acento; las líneas base apagadas para que no compitan
    series_colors = {
        "actual": _hex("#3D3936"),       # barras de ventas reales: finas y tenues
        "lstm": accent,
        "moving_average": _hex("#B09A74"),  # arena apagada (línea discontinua)
        "hist_mean": _hex("#7E9C89"),       # salvia apagada (línea fina)
        "naive": _hex("#77716B"),
    }

    # Tipografía en px LÓGICOS (mínimo 14 para leerse al compartir pantalla)
    size_hero = 84          # portada del menú (serif)
    size_headline = 34      # titular de pantalla (serif)
    size_number = 30        # números grandes (serif)
    size_title = 26
    size_heading = 17
    size_body = 15
    size_label = 14
    size_caption = 14
    tracking = 2            # espaciado entre letras de las etiquetas en mayúsculas

    # Espaciado y forma en px lógicos
    margin = 32
    pad_xs, pad_sm, pad_md, pad_lg, pad_xl = 4, 8, 12, 16, 24
    radius = 6
    radius_sm = 4

    # Compatibilidad con pantallas que aún usen superficies tipo tarjeta
    surface = (255, 255, 255, 8)
    surface_hover = (255, 255, 255, 14)
    surface_alt = (255, 255, 255, 6)
    border = hairline_color
    shadow = (0, 0, 0)


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
    if not sizes:
        return 10_000, 10_000
    w, h = sizes[0]
    if sys.platform == "darwin":  # en Mac el tamaño incluye la barra de menús y el Dock
        h -= 80
    return w, h


def create_window(logical_size, caption: str = "UrbanStyle") -> pygame.Surface:
    """Crea la ventana en píxeles físicos (logical_size × escala DPI × config.UI_ZOOM) y fija THEME.scale.

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
        scale = round(_system_scale() * config.UI_ZOOM, 2)
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
    """Fuente del tamaño (px lógicos) y peso pedidos: regular, medium, semibold, bold (Inter),
    serif o serif_italic (DM Serif Display)."""
    weight = weight or ("bold" if bold else "regular")
    key = (px(size), weight)
    if key not in _font_cache:
        if not _font_cache:  # primera fuente de esta sesión de pygame
            pygame.register_quit(_clear_caches)
        path = FONT_DIR / FONT_FILES[weight]
        _font_cache[key] = pygame.font.Font(str(path) if path.exists() else None, px(size))
    return _font_cache[key]


def _to_display_format(surf: pygame.Surface) -> pygame.Surface:
    """Convierte al formato de la ventana (si ya existe): los blits son varias veces más rápidos."""
    if pygame.display.get_surface() is None:
        return surf
    return surf.convert_alpha() if surf.get_flags() & pygame.SRCALPHA else surf.convert()


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


def render_tracked(text: str, size: int = THEME.size_caption, color=THEME.text_faint,
                   tracking: float = THEME.tracking, weight: str = "semibold") -> pygame.Surface:
    """Texto con espaciado entre letras (pygame no lo trae): para etiquetas en MAYÚSCULAS."""
    key = ("tracked", text, size, color, tracking, weight, THEME.scale)
    surf = _text_cache.get(key)
    if surf is None:
        glyphs = [font(size, weight=weight).render(ch, True, color) for ch in text]
        gap = px(tracking)
        width = sum(g.get_width() for g in glyphs) + gap * max(0, len(glyphs) - 1)
        surf = pygame.Surface((max(1, width), font(size, weight=weight).get_height()), pygame.SRCALPHA)
        x = 0
        for g in glyphs:
            surf.blit(g, (x, 0))
            x += g.get_width() + gap
        surf = _to_display_format(surf)
        _text_cache[key] = surf
    return surf


def blit_eyebrow(surface, text, pos, color=THEME.text_faint, anchor="topleft", size=THEME.size_caption):
    """Etiqueta pequeña en mayúsculas espaciadas (estilo revista). Devuelve su Rect."""
    img = render_tracked(text.upper(), size, color)
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


def fit_size(text: str, max_size: int, min_size: int, width: int, weight: str = "serif") -> int:
    """Mayor tamaño (px lógicos) entre min y max con el que `text` cabe en `width` px físicos."""
    size = max_size
    while size > min_size and font(size, weight=weight).size(text)[0] > width:
        size -= 1
    return size


# ---------------------------------------------------------------------------
# Superficies pre-renderizadas (cacheadas por tamaño)
# ---------------------------------------------------------------------------

def _cached(key, build):
    if key not in _surface_cache:
        _surface_cache[key] = _to_display_format(build())
    return _surface_cache[key]


def rounded_surface(size, radius: int, fill, border=None, border_width: int = 1) -> pygame.Surface:
    """Rectángulo (redondeado o no) con bordes suavizados (se dibuja a 3× y se reduce). Colores RGBA."""
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
    """Fondo casi negro cálido, liso (el diseño editorial no usa resplandores)."""
    w, h = int(size[0]), int(size[1])

    def build():
        surf = pygame.Surface((w, h))
        surf.fill(THEME.bg)
        return surf

    return _cached(("bg", w, h), build)


def hairline(surface, x1: int, x2: int, y: int, color=THEME.hairline_color) -> None:
    """Línea horizontal fina (1 px físico mínimo, baja opacidad) para separar secciones."""
    surface.blit(rounded_surface((max(1, x2 - x1), max(1, px(1))), 0, color), (x1, y))


def vhairline(surface, x: int, y1: int, y2: int, color=THEME.hairline_color) -> None:
    surface.blit(rounded_surface((max(1, px(1)), max(1, y2 - y1)), 0, color), (x, y1))


def draw_card(surface, rect, radius=None, fill=THEME.surface, border=THEME.border, shadow: bool = False,
              elevation: float = 0.0) -> None:
    """Compatibilidad: panel translúcido plano (el lenguaje editorial evita las tarjetas; úsalo poco)."""
    rect = pygame.Rect(rect)
    radius = px(THEME.radius) if radius is None else radius
    surface.blit(rounded_surface(rect.size, radius, fill, border, max(1, px(1))), rect)


def pill(surface, text, pos, color=THEME.accent, text_color=None, size=THEME.size_caption, anchor="topleft",
         weight="semibold"):
    """Compatibilidad: etiqueta en mayúsculas espaciadas del color dado (ya sin fondo de píldora)."""
    img = render_tracked(text.upper(), size, text_color or color, weight=weight)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


# ---------------------------------------------------------------------------
# Dibujo con antialias
# ---------------------------------------------------------------------------

def aa_circle(surface, color, center, radius) -> None:
    x, y, r = int(round(center[0])), int(round(center[1])), int(round(radius))
    pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    pygame.gfxdraw.aacircle(surface, x, y, r, color)


def aa_line(surface, color, p1, p2, width: int = 1) -> None:
    """Línea suavizada de cualquier grosor (cuadrilátero con bordes antialias)."""
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


def _mix(a, b, t: float) -> tuple:
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class Button:
    """Botón editorial (sin caja). API: Button(rect, label, font=None), draw(surface), handle_event(event) -> bool.

    * Por defecto, enlace de texto: gris que pasa a blanco con un subrayado de acento al pasar el mouse.
    * `primary=True`: enlace en el color de acento.
    * Con `subtitle` (y opcional `badge`): fila de una lista tipográfica (letra en serif, nombre grande en
      serif, descripción pequeña); al pasar el mouse se tiñe del acento y se desplaza un poco a la derecha.
    `rect` es la zona clicable en píxeles físicos (usa ui.px).
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
        if self.subtitle:
            self._draw_list_row(surface, h)
            return
        base = THEME.accent_text if self.primary else THEME.text_muted
        color = _mix(base, THEME.text if not self.primary else THEME.accent, h)
        if self.font is not None:  # compatibilidad: fuente propia
            img = self.font.render(self.label, True, color)
            rect = img.get_rect(center=self.rect.center)
            surface.blit(img, rect)
        else:
            rect = blit_text(surface, self.label, self.rect.center, THEME.size_body, color, weight="medium",
                             anchor="center")
        self.draw_rect = rect
        underline = int(rect.width * (h if not self.primary else 0.35 + 0.65 * h))
        if underline > 0:
            pygame.draw.rect(surface, THEME.accent, (rect.left, rect.bottom + px(2), underline, max(1, px(1.5))))

    def _draw_list_row(self, surface, h: float) -> None:
        shift = int(round(px(8) * h))
        rect = self.draw_rect = self.rect.move(shift, 0)
        x = rect.left
        if self.badge:
            blit_text(surface, self.badge, (x, rect.centery), THEME.size_title + 6,
                      _mix(THEME.text_faint, THEME.accent, h), weight="serif", anchor="midleft")
            x += px(44)
        size = fit_size(self.label, THEME.size_title + 4, THEME.size_heading, self.rect.right - x - px(8))
        blit_text(surface, self.label, (x, rect.centery + px(2)), size,
                  _mix(THEME.text, THEME.accent, h), weight="serif", anchor="bottomleft")
        blit_text(surface, self.subtitle, (x, rect.centery + px(6)), THEME.size_label, THEME.text_faint)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """True si se hizo clic izquierdo sobre el botón."""
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


class Slider:
    """Slider horizontal fino. API: Slider(rect, min_value, max_value, value), handle_event(event) -> valor,
    draw(surface), x_of(valor).

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
        return self.rect.inflate(px(28), px(28)).collidepoint(pos)

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

    def draw(self, surface: pygame.Surface, stops: int = 0) -> None:
        """Riel de 1 px, marcas opcionales en `stops` posiciones enteras y control circular del acento."""
        y = self.rect.centery
        hairline(surface, self.rect.left, self.rect.right, y, (*THEME.text, 70))
        for i in range(stops):
            x = self.x_of(self.min_value + i * (self.max_value - self.min_value) / max(1, stops - 1))
            vhairline(surface, x, y - px(4), y + px(5), (*THEME.text, 90))
        x = self.x_of(self.value)
        h = self._hover.step(self.dragging or self._hit(pygame.mouse.get_pos()))
        aa_circle(surface, THEME.accent, (x, y), px(6 + 2 * h))
        aa_circle(surface, THEME.bg, (x, y), px(2.5))


class Toast:
    """Aviso temporal sobrio (fondo sólido, filete de acento). show(texto), draw(surface) en cada cuadro."""

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
        box = img.get_rect().inflate(px(44), px(22))
        box.midbottom = (surface.get_width() // 2, surface.get_height() - px(28))
        layer = pygame.Surface(box.size, pygame.SRCALPHA)
        layer.fill((*THEME.bg_raised, 250))
        pygame.draw.rect(layer, THEME.accent, (0, 0, max(2, px(3)), box.height))
        layer.blit(img, img.get_rect(midleft=(px(22), box.height // 2)))
        layer.set_alpha(255 if remaining > 400 else int(255 * remaining / 400))  # se desvanece al final
        surface.blit(layer, box)


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
