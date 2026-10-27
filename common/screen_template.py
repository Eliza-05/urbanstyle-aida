"""Plantilla de pantalla en el estilo editorial de UrbanStyle: cópiala como punto de partida para A o C.

Cómo usarla:
  1. Copia este archivo a tu carpeta (p. ej. `task_a_pathfinding/screen.py`) y cambia EYEBROW, HEADLINE y
     las funciones `draw_main` / `on_button` / `on_slider`.
  2. Conserva el contrato con el menú: `run(screen, clock)` corre su propio bucle y vuelve con ESC o con el
     botón "← Menú". Si cierran la ventana, reenvía QUIT para que el menú también se cierre.
  3. Separa la lógica (grid, búsqueda, entorno, agente) de esta clase: aquí solo se dibuja y se leen eventos.
     Los modelos entrenados se CARGAN; nunca se entrena dentro de Pygame.

Qué trae: cabecera (etiqueta + titular serif + línea fina), zona principal con el fondo sutil de gráfica,
un panel derecho con un slider y un botón de ejemplo, un tooltip al pasar el mouse sobre la zona principal,
un aviso (Toast) y aparición suave. Geometría en píxeles lógicos (ventana base 1024×640) con `ui.px()`.
"""
import pygame

from common import config, ui
from common.ui import THEME, px

EYEBROW = "Plantilla · Pantalla de ejemplo"
HEADLINE = "Un titular que dice lo que muestra la pantalla"
SAMPLE = [3, 5, 4, 8, 6, 9, 7, 11, 10, 12]   # datos de ejemplo para la zona principal


def R(x, y, w, h) -> pygame.Rect:
    """Rect en píxeles físicos a partir de coordenadas lógicas."""
    return pygame.Rect(px(x), px(y), px(w), px(h))


class TemplateScreen:
    """Estado e interacción de la pantalla. Métodos: handle_event, update(dt), draw(surface)."""

    def __init__(self, size=None):
        self.size = size or (px(config.WINDOW_WIDTH), px(config.WINDOW_HEIGHT))
        self.done = False                       # True -> run() vuelve al menú
        self.mouse = (-1, -1)
        self.clicks = 0
        self.toast, self.fade = ui.Toast(), ui.FadeIn()

        m = THEME.margin
        w, h = self.size[0] / THEME.scale, self.size[1] / THEME.scale   # tamaño lógico
        self.main = R(m, 130, w - 2 * m - 280, h - 130 - 70)            # zona principal (la protagonista)
        panel_x = w - m - 232
        self.panel = R(panel_x, 130, 232, h - 130 - 70)
        self.slider = ui.Slider(R(panel_x, 200, 232, 20), 1, 10, 5)
        self.button = ui.Button(R(panel_x, 260, 64, 30), "Ejemplo", primary=True)
        self.back_button = ui.Button(R(m, h - 52, 80, 30), "← Menú")
        self.static = ui._to_display_format(self.render_static())

    # ---------- lógica de interacción ----------
    def on_button(self) -> None:
        self.clicks += 1
        self.toast.show(f"Clic número {self.clicks} · valor del slider: {round(self.slider.value)}")

    def on_slider(self, value: float) -> None:
        """Se llama en cada cambio del slider; aquí reacciona tu pantalla (p. ej. nº de clientes)."""

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or self.back_button.handle_event(event):
            self.done = True
            return
        before = self.slider.value
        if self.slider.handle_event(event) != before:
            self.on_slider(self.slider.value)
        if self.button.handle_event(event):
            self.on_button()

    def update(self, dt: float) -> None:
        """Avanza la simulación `dt` segundos (la velocidad no debe depender de los FPS)."""

    # ---------- dibujo ----------
    def render_static(self) -> pygame.Surface:
        """Lo que no cambia entre cuadros se pinta una sola vez."""
        layer = ui.background(self.size).copy()
        m = px(THEME.margin)
        ui.blit_eyebrow(layer, EYEBROW, (m, px(22)), THEME.accent_text)
        size = ui.fit_size(HEADLINE, THEME.size_headline, THEME.size_title, self.size[0] - 2 * m)
        ui.blit_text(layer, HEADLINE, (m, px(46)), size, weight="serif")
        ui.hairline(layer, m, self.size[0] - m, px(104))
        layer.blit(ui.rounded_surface(self.main.size, px(THEME.radius), THEME.chart_bg), self.main)
        ui.vhairline(layer, self.panel.left - px(24), self.panel.top, self.panel.bottom)
        ui.blit_eyebrow(layer, "Parámetro", (self.panel.left, self.panel.top))
        ui.blit_eyebrow(layer, "Acción", (self.panel.left, px(236)))
        return layer

    def draw_main(self, surface: pygame.Surface) -> None:
        """Zona principal: aquí va tu grid, tu gráfica o tu simulación."""
        r = self.main.inflate(-px(40), -px(40))
        top = max(SAMPLE)
        points = [(r.left + r.width * i / (len(SAMPLE) - 1), r.bottom - r.height * v / top)
                  for i, v in enumerate(SAMPLE)]
        for i in range(1, 4):                                           # rejilla muy tenue
            ui.hairline(surface, r.left, r.right, r.bottom - r.height * i // 4, THEME.grid)
        ui.aa_polyline(surface, THEME.accent, points, px(2))
        for p in points:
            ui.aa_circle(surface, THEME.accent, p, px(3))

    def hovered_index(self) -> int | None:
        r = self.main.inflate(-px(40), -px(40))
        if not r.inflate(px(20), px(20)).collidepoint(self.mouse):
            return None
        i = round((self.mouse[0] - r.left) / r.width * (len(SAMPLE) - 1))
        return min(max(i, 0), len(SAMPLE) - 1)

    def draw_tooltip(self, surface: pygame.Surface, i: int) -> None:
        """Tooltip discreto: fondo sólido sin sombra y un filete fino (mismo recurso que la Tarea B)."""
        title = ui.render_text(f"Punto {i + 1}", THEME.size_label, THEME.text, weight="semibold")
        value = ui.render_text(f"valor {SAMPLE[i]}", THEME.size_label, THEME.text_muted)
        box = pygame.Rect(0, 0, max(title.get_width(), value.get_width()) + px(28), px(66))
        box.topleft = (self.mouse[0] + px(14), self.mouse[1] - box.height // 2)
        box.clamp_ip(self.main)
        pygame.draw.rect(surface, THEME.bg_raised, box, border_radius=px(THEME.radius))
        surface.blit(ui.rounded_surface(box.size, px(THEME.radius), (*THEME.bg_raised, 0), THEME.hairline_color), box)
        surface.blit(title, (box.left + px(14), box.top + px(10)))
        surface.blit(value, (box.left + px(14), box.top + px(34)))

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.static, (0, 0))
        self.draw_main(surface)
        ui.blit_text(surface, str(round(self.slider.value)), (self.panel.left, px(165)), THEME.size_number + 4,
                     weight="serif", anchor="midleft")
        self.slider.draw(surface)
        self.button.draw(surface)
        self.back_button.draw(surface)
        hover = self.hovered_index()
        if hover is not None:
            self.draw_tooltip(surface, hover)
        self.toast.draw(surface)
        self.fade.draw(surface)


def run(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    """Contrato con el menú: bucle propio; vuelve con ESC / "← Menú"."""
    view = TemplateScreen(size=screen.get_size())
    while not view.done:
        dt = clock.tick(config.FPS) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.event.post(pygame.event.Event(pygame.QUIT))  # el menú también debe cerrarse
                return
            view.handle_event(event)
        view.update(dt)
        view.draw(screen)
        pygame.display.flip()
