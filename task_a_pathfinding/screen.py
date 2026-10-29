"""Tarea A (Sebastián): pantalla Pygame de rutas de clientes y tráfico en la tienda (lenguaje editorial).

La grilla es la protagonista: secciones, clientes en movimiento y, encima, el mapa de calor de visitas por
celda con las zonas de alto tráfico marcadas. A la derecha, los controles (algoritmo, distribución, número
de clientes, velocidad, capas) y las medidas de la simulación en curso.
Toda la lógica está en store.py / search.py / simulation.py; aquí solo se dibuja y se leen eventos.
Geometría en píxeles lógicos (ventana base 1024×640) convertida con ui.px().
"""
import pygame

from common import config, ui
from common.ui import THEME, px
from task_a_pathfinding.search import ALGORITHMS
from task_a_pathfinding.simulation import Simulation, load_profiles
from task_a_pathfinding.store import CELL_NAMES, LAYOUTS, SECTIONS, WALL, load_layout

EYEBROW = "Tarea A · Rutas de clientes y layout"
SECTION_COLORS = {"F": (196, 128, 146), "M": (110, 146, 178), "B": THEME.warning, "T": THEME.success}
SHELF_COLOR = (58, 53, 49)          # estanterías interiores (las paredes del borde no se dibujan)
FLOOR_COLOR = (255, 248, 240, 16)
DEFAULT_CUSTOMERS, MIN_CUSTOMERS, MAX_CUSTOMERS = 40, 5, 100
DEFAULT_SPEED = 2.0          # segundos simulados por segundo real
LAYERS = ("Calor", "Rutas", "Búsqueda")


def R(x, y, w, h) -> pygame.Rect:
    """Rect en píxeles físicos a partir de coordenadas lógicas."""
    return pygame.Rect(px(x), px(y), px(w), px(h))


def fmt(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def word_row(labels, x: float, y: float, gap: float = 18) -> list[ui.Button]:
    """Fila de enlaces de texto (x, y lógicos): el ancho de cada uno sale de su texto."""
    buttons = []
    for label in labels:
        w = ui.font(THEME.size_body, weight="medium").size(label)[0] / THEME.scale + 8
        buttons.append(ui.Button(R(x, y, w, 26), label))
        x += w + gap
    return buttons


def solid_box(surface, box: pygame.Rect) -> None:
    """Fondo de tooltip: sólido y sin sombra, con un filete fino (mismo recurso que las Tareas B y D)."""
    pygame.draw.rect(surface, THEME.bg_raised, box, border_radius=px(THEME.radius))
    surface.blit(ui.rounded_surface(box.size, px(THEME.radius), (*THEME.bg_raised, 0), THEME.hairline_color), box)


class PathfindingScreen:
    """Estado e interacción de la pantalla. Métodos: handle_event, update(dt), draw(surface)."""

    def __init__(self, size=None, profiles=None):
        self.size = size or (px(config.WINDOW_WIDTH), px(config.WINDOW_HEIGHT))
        self.done = False
        self.mouse = (-1, -1)
        self.toast, self.fade = ui.Toast(), ui.FadeIn()
        self.profiles = profiles or load_profiles()
        self.algorithm, self.layout_name = "A*", "Pasillos"
        self.seed = config.SEED
        self.paused = False
        self.layers = {"Calor": True, "Rutas": False, "Búsqueda": False}
        self._cursor = None

        W, H = self.size[0] / THEME.scale, self.size[1] / THEME.scale
        m = THEME.margin
        back_w = ui.font(THEME.size_body, weight="medium").size("← Menú")[0] / THEME.scale + 12
        self.back_button = ui.Button(R(W - m - back_w, 14, back_w, 30), "← Menú")
        # Panel derecho (controles y medidas).
        self.px0, self.pw = W - m - 236, 236
        x = self.px0
        self.algo_buttons = word_row(ALGORITHMS, x, 154)
        self.layout_buttons = word_row(LAYOUTS, x, 216, gap=14)
        self.customers_slider = ui.Slider(R(x, 284, self.pw, 20), MIN_CUSTOMERS, MAX_CUSTOMERS, DEFAULT_CUSTOMERS)
        self.speed_slider = ui.Slider(R(x, 342, self.pw, 20), 0.5, 4.0, DEFAULT_SPEED)
        self.action_buttons = word_row(("Reiniciar", "Pausa", "Otra muestra"), x, 376, gap=10)
        self.layer_buttons = word_row(LAYERS, x, 406, gap=14)
        # Grilla: celdas cuadradas a la izquierda del panel. Las paredes del borde no se dibujan, así que la
        # grilla se corre una celda para que el contorno de la tienda quede alineado con el margen; solo la
        # fila de abajo (entrada y salida) ocupa espacio.
        self.store = load_layout(self.layout_name)
        area_w, area_h = self.px0 - 28 - m, 424
        self.cell = int(min(area_w / (self.store.n_cols - 2), area_h / (self.store.n_rows - 1)))
        self.grid = R(m - self.cell, 140 - self.cell, self.cell * self.store.n_cols, self.cell * self.store.n_rows)
        self.static = None
        self.restart()

    # ---------- simulación ----------
    def restart(self, new_seed: bool = False) -> None:
        if new_seed:
            self.seed += 1
        self.store = load_layout(self.layout_name)
        self.sim = Simulation(self.store, self.algorithm, round(self.customers_slider.value), seed=self.seed,
                              profiles=self.profiles)
        self.paused = False
        self.action_buttons[1].label = "Pausa"
        for b in self.algo_buttons:
            b.primary = b.label == self.algorithm
        for b in self.layout_buttons:
            b.primary = b.label == self.layout_name
        for b in self.layer_buttons:
            b.primary = self.layers[b.label]
        self.static = ui._to_display_format(self.render_static())

    def cell_rect(self, r: int, c: int) -> pygame.Rect:
        return pygame.Rect(self.grid.left + c * px(self.cell), self.grid.top + r * px(self.cell),
                           px(self.cell), px(self.cell))

    def cell_center(self, r: float, c: float) -> tuple[float, float]:
        return (self.grid.left + (c + 0.5) * px(self.cell), self.grid.top + (r + 0.5) * px(self.cell))

    def hovered_cell(self):
        if not self.grid.collidepoint(self.mouse):
            return None
        c = (self.mouse[0] - self.grid.left) // px(self.cell)
        r = (self.mouse[1] - self.grid.top) // px(self.cell)
        return (int(r), int(c)) if self.store.in_bounds((r, c)) else None

    # ---------- eventos ----------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or self.back_button.handle_event(event):
            self.done = True
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.toggle_pause()
            return
        for b in self.algo_buttons:
            if b.handle_event(event) and b.label != self.algorithm:
                self.algorithm = b.label
                self.restart()
                self.toast.show(f"{b.label}: mismos clientes, rutas planeadas de nuevo")
        for b in self.layout_buttons:
            if b.handle_event(event) and b.label != self.layout_name:
                self.layout_name = b.label
                self.restart()
        reset, pause, reseed = self.action_buttons
        if reset.handle_event(event):
            self.restart()
        if pause.handle_event(event):
            self.toggle_pause()
        if reseed.handle_event(event):
            self.restart(new_seed=True)
            self.toast.show(f"Otros clientes del dataset (semilla {self.seed})")
        for b in self.layer_buttons:
            if b.handle_event(event):
                self.layers[b.label] = not self.layers[b.label]
                b.primary = self.layers[b.label]
        was_dragging = self.customers_slider.dragging
        self.customers_slider.handle_event(event)
        if was_dragging and not self.customers_slider.dragging:  # al soltar: simulación nueva
            self.customers_slider.value = round(self.customers_slider.value)
            if self.customers_slider.value != len(self.sim.customers):
                self.restart()
        self.speed_slider.handle_event(event)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.action_buttons[1].label = "Seguir" if self.paused else "Pausa"

    def update(self, dt: float) -> None:
        if not self.paused and not self.sim.finished:
            self.sim.step(min(dt, 0.1) * self.speed_slider.value)  # sin saltos grandes si la ventana se congela

    # ---------- dibujo ----------
    def render_static(self) -> pygame.Surface:
        """Fondo, cabecera fija, rótulos del panel y la grilla del layout actual (se rehace al cambiar layout)."""
        layer = ui.background(self.size).copy()
        m = px(THEME.margin)
        ui.blit_eyebrow(layer, EYEBROW, (m, px(22)), THEME.accent_text)
        ui.blit_text(layer, f"Tienda de {self.store.n_cols} × {self.store.n_rows} celdas  ·  cada cliente es una "
                     "transacción real del dataset (categoría, género y cantidad)", (m, px(94)), THEME.size_label,
                     THEME.text_muted)
        ui.hairline(layer, m, self.size[0] - m, px(120))
        x = px(self.px0)
        ui.vhairline(layer, x - px(24), px(134), self.grid.bottom - px(self.cell))
        ui.blit_eyebrow(layer, "Algoritmo", (x, px(134)))
        ui.blit_eyebrow(layer, "Distribución", (x, px(196)))
        ui.blit_eyebrow(layer, "Clientes", (x, px(254)))
        ui.blit_eyebrow(layer, "Velocidad", (x, px(312)))
        ui.hairline(layer, x, x + px(self.pw), px(440))
        # Grilla: contorno de la tienda, pasillos claros y estanterías como bloques sólidos.
        last_r, last_c = self.store.n_rows - 1, self.store.n_cols - 1
        inner = self.cell_rect(1, 1).union(self.cell_rect(last_r - 1, last_c - 1))
        pygame.draw.rect(layer, (*THEME.text_faint,), inner.inflate(px(6), px(6)), max(1, px(1)),
                         border_radius=px(THEME.radius))
        for r in range(self.store.n_rows):
            for c in range(self.store.n_cols):
                rect = self.cell_rect(r, c)
                kind = self.store.kind((r, c))
                if kind == WALL:
                    if 0 < r < last_r and 0 < c < last_c:
                        pygame.draw.rect(layer, SHELF_COLOR, rect.inflate(-px(6), -px(6)),
                                         border_radius=px(THEME.radius_sm))
                    continue
                fill = (*SECTION_COLORS[kind], 46) if kind in SECTIONS else FLOOR_COLOR
                if kind == "C":
                    fill = (*THEME.text, 34)
                layer.blit(ui.rounded_surface(rect.inflate(-max(1, px(1)), -max(1, px(1))).size, 0, fill),
                           rect.inflate(-max(1, px(1)), -max(1, px(1))))
        for kind, cells in [*self.store.sections.items(), ("C", self.store.checkouts)]:
            rects = [self.cell_rect(*cell) for cell in cells]
            box = rects[0].unionall(rects[1:])
            label = CELL_NAMES[kind]
            color = SECTION_COLORS.get(kind, THEME.text)
            if ui.font(THEME.size_label, weight="semibold").size(label)[0] > box.width - px(6):
                label = label.split()[0]
            ui.blit_text(layer, label, box.center, THEME.size_label, color, weight="semibold", anchor="center")
        for cell, label in ((self.store.entrance, "Entrada ↑"), (self.store.exit, "Salida ↓")):
            rect = self.cell_rect(*cell)
            ui.blit_text(layer, label, (rect.centerx, rect.bottom + px(4)), THEME.size_caption, THEME.text_muted,
                         anchor="midtop")
            pygame.draw.rect(layer, THEME.text_muted, rect.inflate(-px(10), -px(10)), max(1, px(1.5)),
                             border_radius=px(THEME.radius_sm))
        # Leyenda de secciones.
        lx, ly = m, self.grid.bottom + px(30)
        for letter, (name, cat, _) in SECTIONS.items():
            ui.aa_circle(layer, SECTION_COLORS[letter], (lx + px(5), ly + px(9)), px(5))
            r = ui.blit_text(layer, f"{name}", (lx + px(16), ly), THEME.size_caption, THEME.text_muted)
            lx = r.right + px(18)
        pygame.draw.rect(layer, THEME.accent, (lx, ly + px(3), px(12), px(12)), max(1, px(1.5)))
        ui.blit_text(layer, "zona de alto tráfico (10 % superior)", (lx + px(18), ly), THEME.size_caption,
                     THEME.text_muted)
        return layer

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.static, (0, 0))
        self.draw_header(surface)
        if self.layers["Calor"]:
            self.draw_heat(surface)
        if self.layers["Búsqueda"]:
            self.draw_search(surface)
        if self.layers["Rutas"]:
            self.draw_routes(surface)
        self.draw_customers(surface)
        self.draw_panel(surface)
        hover = self.hovered_cell()
        if hover is not None and self.store.passable(hover):
            self.draw_tooltip(surface, hover)
        self.back_button.draw(surface)
        self.toast.draw(surface)
        self.update_cursor()
        self.fade.draw(surface)

    def draw_header(self, surface) -> None:
        m = px(THEME.margin)
        hot = self.sim.hotspots(1, aisles_only=True)
        if hot:
            place = self.store.describe(hot[0][0])
            text = f"Más tránsito: el {place[0].lower()}{place[1:]}"
        else:
            text = "¿Dónde se aglomeran los clientes?"
        size = ui.fit_size(text, THEME.size_headline, THEME.size_title, px(self.px0 - 60) - m)
        ui.blit_text(surface, text, (m, px(84)), size, weight="serif", anchor="bottomleft")

    def draw_heat(self, surface) -> None:
        visits = self.sim.visits
        top = visits.max()
        if top == 0:
            return
        hot = self.sim.high_traffic_mask()
        layer = pygame.Surface(self.grid.size, pygame.SRCALPHA)
        for r, c in self.store.walkable:
            v = visits[r, c]
            if v:
                rect = self.cell_rect(r, c).move(-self.grid.left, -self.grid.top)
                layer.fill((*THEME.accent, int(25 + 160 * (v / top) ** 0.8)), rect)
        surface.blit(layer, self.grid)
        for r, c in self.store.walkable:
            if hot[r, c]:
                pygame.draw.rect(surface, THEME.accent, self.cell_rect(r, c).inflate(-px(3), -px(3)),
                                 max(1, px(1.5)))

    def draw_search(self, surface) -> None:
        """Celdas que expandió la última búsqueda (más claras = expandidas antes) y su ruta."""
        if self.sim.last_search is None:
            return
        start, goal, result = self.sim.last_search
        n = max(1, len(result.explored))
        for i, cell in enumerate(result.explored):
            shade = int(235 - 140 * i / n)
            ui.aa_circle(surface, (shade, shade - 4, shade - 10), self.cell_center(*cell), px(self.cell * 0.11))
        if result.path:
            ui.aa_polyline(surface, THEME.text, [self.cell_center(*cell) for cell in result.path], px(2))
        for cell in (start, goal):
            ui.aa_circle(surface, THEME.text, self.cell_center(*cell), px(4))

    def draw_routes(self, surface) -> None:
        """Lo que le falta por recorrer a cada cliente dentro de la tienda."""
        layer = pygame.Surface(self.grid.size, pygame.SRCALPHA)
        ox, oy = self.grid.topleft
        for c in self.sim.customers:
            if not c.active:
                continue
            i = int(c.pos)
            pts = [self.cell_center(*c.xy())] + [self.cell_center(*cell) for cell in c.path[i + 1:]]
            if len(pts) > 1:
                pygame.draw.lines(layer, (*SECTION_COLORS[c.profile.section], 110), False,
                                  [(x - ox, y - oy) for x, y in pts], max(1, px(1.5)))
        surface.blit(layer, self.grid)

    def draw_customers(self, surface) -> None:
        radius = max(3, px(self.cell * 0.16))
        spread = self.cell * 0.2
        for c in self.sim.customers:
            if not c.active:
                continue
            r, col = c.xy()
            x, y = self.cell_center(r, col)
            x += px(((c.id * 37) % 7 - 3) / 3 * spread)   # desfase fijo por cliente: no se tapan del todo
            y += px(((c.id * 53) % 7 - 3) / 3 * spread)
            ui.aa_circle(surface, THEME.bg, (x, y), radius + px(1.5))
            ui.aa_circle(surface, SECTION_COLORS[c.profile.section], (x, y), radius)

    def draw_panel(self, surface) -> None:
        x, right = px(self.px0), px(self.px0 + self.pw)
        for b in (*self.algo_buttons, *self.layout_buttons, *self.action_buttons, *self.layer_buttons):
            b.draw(surface)
        ui.blit_text(surface, str(round(self.customers_slider.value)), (right, px(250)), THEME.size_title,
                     weight="serif", anchor="topright")
        ui.blit_text(surface, f"{fmt(self.speed_slider.value)}×", (right, px(308)), THEME.size_title,
                     weight="serif", anchor="topright")
        self.customers_slider.draw(surface)
        self.speed_slider.draw(surface)

        st = self.sim.stats()
        hot = self.sim.hotspots(1)
        rows = [
            ("Atendidos", f"{st['served']} de {st['customers']}"
                          + (f"  ·  {st['in_store']} dentro" if st["in_store"] else "")),
            ("Ruta media", f"{fmt(st['mean_route'])} pasos" if st["mean_route"] else "—"),
            (f"Nodos expandidos ({self.algorithm})", fmt(st["mean_expanded"]) if st["searches"] else "—"),
            ("Tiempo por búsqueda", f"{fmt(st['mean_search_us'], 0)} µs" if st["searches"] else "—"),
            ("Celda más visitada", f"{hot[0][1]} visitas" if hot else "—"),
            ("Pico en una celda", f"{st['max_peak']} clientes"),
        ]
        y = px(450)
        for label, value in rows:
            ui.blit_text(surface, label, (x, y), THEME.size_label, THEME.text_faint)
            ui.blit_text(surface, value, (right, y), THEME.size_label, THEME.text, weight="semibold",
                         anchor="topright")
            y += px(21)
        state = "Simulación terminada" if self.sim.finished else ("En pausa" if self.paused else "")
        clock = f"t = {fmt(st['time'], 0)} s simulados" + (f"  ·  {state}" if state else "")
        ui.blit_text(surface, clock, (x, self.grid.bottom + px(30)), THEME.size_caption,
                     THEME.accent_text if self.sim.finished else THEME.text_faint)

    def draw_tooltip(self, surface, cell) -> None:
        r, c = cell
        title = ui.render_text(self.store.describe(cell), THEME.size_label, THEME.text, weight="semibold")
        rows = [("Visitas", str(int(self.sim.visits[r, c]))),
                ("Ocupación", f"{fmt(self.sim.occupancy[r, c])} s"),
                ("Pico simultáneo", str(int(self.sim.peak[r, c])))]
        box = pygame.Rect(0, 0, max(title.get_width(), px(170)) + px(28), px(38) + px(21) * len(rows))
        box.topleft = (self.mouse[0] + px(16), self.mouse[1] - box.height // 2)
        box.clamp_ip(surface.get_rect().inflate(-px(8), -px(8)))
        solid_box(surface, box)
        surface.blit(title, (box.left + px(14), box.top + px(10)))
        y = box.top + px(36)
        for label, value in rows:
            ui.blit_text(surface, label, (box.left + px(14), y), THEME.size_label, THEME.text_muted)
            ui.blit_text(surface, value, (box.right - px(14), y), THEME.size_label, THEME.text,
                         weight="semibold", anchor="topright")
            y += px(21)

    def update_cursor(self) -> None:
        """Cursor de mano sobre lo que se puede pulsar."""
        buttons = (*self.algo_buttons, *self.layout_buttons, *self.action_buttons, *self.layer_buttons,
                   self.back_button)
        hand = any(b.rect.collidepoint(self.mouse) for b in buttons) or self.customers_slider.dragging \
            or self.speed_slider.dragging
        if hand != self._cursor:
            self._cursor = hand
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hand else pygame.SYSTEM_CURSOR_ARROW)
            except pygame.error:  # sin ventana real (p. ej. pruebas headless)
                pass


def run(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    """Contrato con el menú: bucle propio; vuelve con ESC / "← Menú"."""
    view = PathfindingScreen(size=screen.get_size())
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
