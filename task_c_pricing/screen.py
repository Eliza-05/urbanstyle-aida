"""Tarea C (Sebastián): pantalla Pygame de precios dinámicos con RL (lenguaje editorial).

Tres políticas viven el MISMO episodio a la vez (misma semilla = mismos clientes): el agente entrenado
(PPO o DQN), el mejor precio fijo y el oráculo. Arriba, el ingreso acumulado en vivo (la protagonista);
abajo, el precio que cobra el agente día a día con el segmento de cada día. Controles: agente, uno o tres
productos, precio inicial, velocidad, reiniciar, pausa y otra semilla.
Solo CARGA los agentes de models/ (task_c_pricing.train); nunca entrena.
Geometría en píxeles lógicos (ventana base 1024×640) convertida con ui.px().
"""
import json

import numpy as np
import pygame

from common import config, ui
from common.ui import THEME, px
from task_c_pricing.demand import PRODUCT_NAMES, PRODUCTS, SEGMENT_NAMES, load_config
from task_c_pricing.env import EPISODE_DAYS, PricingEnv
from task_c_pricing.policies import (SUMMARY_JSON, agent_policy, best_fixed_levels, keep_policy, load_agent,
                                     oracle_policy)

EYEBROW = "Tarea C · Precios dinámicos con RL"
POLICY_COLORS = {"agent": THEME.accent, "fixed": THEME.warning, "oracle": THEME.success}
POLICY_LABELS = {"fixed": "Mejor precio fijo", "oracle": "Oráculo"}
SEGMENT_COLORS = [(110, 146, 178), (196, 128, 146), (160, 140, 196)]
PRODUCT_COLORS = [THEME.accent, (224, 176, 140), (140, 176, 206)]
MODES = {"Solo ropa": ("Clothing",), "Tres productos": PRODUCTS}
DEFAULT_SPEED = 6.0          # días por segundo

_cache = {}


def load_bundle() -> dict:
    """Configuración de la demanda, agentes y resumen de la evaluación (una vez por sesión)."""
    if "bundle" not in _cache:
        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8")) if SUMMARY_JSON.exists() else None
        _cache["bundle"] = {"cfg": load_config(), "summary": summary,
                            "agents": {"PPO": load_agent("ppo"), "DQN": load_agent("dqn"),
                                       "PPO multi": load_agent("ppo_multi")}}
    return _cache["bundle"]


def R(x, y, w, h) -> pygame.Rect:
    return pygame.Rect(px(x), px(y), px(w), px(h))


def fmt(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def money(value: float) -> str:
    return "$" + f"{value:,.0f}".replace(",", ".")


def word_row(labels, x: float, y: float, gap: float = 18) -> list[ui.Button]:
    """Fila de enlaces de texto (x, y lógicos): el ancho de cada uno sale de su texto."""
    buttons = []
    for label in labels:
        w = ui.font(THEME.size_body, weight="medium").size(label)[0] / THEME.scale + 8
        buttons.append(ui.Button(R(x, y, w, 26), label))
        x += w + gap
    return buttons


def solid_box(surface, box: pygame.Rect) -> None:
    pygame.draw.rect(surface, THEME.bg_raised, box, border_radius=px(THEME.radius))
    surface.blit(ui.rounded_surface(box.size, px(THEME.radius), (*THEME.bg_raised, 0), THEME.hairline_color), box)


class PricingScreen:
    """Estado e interacción de la pantalla. Métodos: handle_event, update(dt), draw(surface)."""

    def __init__(self, bundle=None, error: str | None = None, size=None):
        self.size = size or (px(config.WINDOW_WIDTH), px(config.WINDOW_HEIGHT))
        self.bundle, self.error = bundle, error
        self.done = False
        self.mouse = (-1, -1)
        self.toast, self.fade = ui.Toast(), ui.FadeIn()
        self._cursor = None
        W, H = self.size[0] / THEME.scale, self.size[1] / THEME.scale
        m = THEME.margin
        back_w = ui.font(THEME.size_body, weight="medium").size("← Menú")[0] / THEME.scale + 12
        self.back_button = ui.Button(R(W - m - back_w, 14, back_w, 30), "← Menú")
        if bundle is None:
            return
        self.cfg = bundle["cfg"]
        self.agent_name, self.mode, self.seed = "PPO", "Solo ropa", config.SEED
        self.paused, self.clock = False, 0.0
        # Panel derecho.
        self.px0, self.pw = W - m - 236, 236
        x = self.px0
        self.agent_buttons = word_row(("PPO", "DQN"), x, 154)
        self.mode_buttons = word_row(MODES, x, 216)
        self.price_slider = ui.Slider(R(x, 284, self.pw, 20), 0, self.cfg.n_levels - 1, 5)
        self.speed_slider = ui.Slider(R(x, 342, self.pw, 20), 1, 20, DEFAULT_SPEED)
        self.action_buttons = word_row(("Reiniciar", "Pausa", "Otra semilla"), x, 376, gap=10)
        # Gráficas: ingreso acumulado (arriba) y precio con el segmento de cada día (abajo).
        chart_right = self.px0 - 28
        self.labels_x = px(chart_right - 112)
        self.revenue_plot = R(m, 150, chart_right - 112 - 16 - m - 44, 196)
        self.price_plot = R(m, 404, self.revenue_plot.width / THEME.scale, 120)
        self.strip_y = self.price_plot.bottom + px(8)
        self.restart()

    # ---------- episodio ----------
    @property
    def products(self) -> tuple[str, ...]:
        return MODES[self.mode]

    @property
    def model_key(self) -> str:
        return "PPO multi" if self.mode == "Tres productos" else self.agent_name

    def restart(self, new_seed: bool = False) -> None:
        if new_seed:
            self.seed += 1
        level = int(round(self.price_slider.value))
        self.price_slider.value = level
        self.envs = {name: PricingEnv(self.cfg, self.products) for name in ("agent", "fixed", "oracle")}
        self.best_fixed = best_fixed_levels(self.envs["fixed"])
        self.policies = {"agent": agent_policy(self.bundle["agents"][self.model_key]), "fixed": keep_policy,
                         "oracle": oracle_policy}
        starts = {"agent": level, "fixed": self.best_fixed, "oracle": level}
        self.obs = {name: env.reset(seed=self.seed, options={"initial_level": starts[name]})[0]
                    for name, env in self.envs.items()}
        self.history = {name: [] for name in self.envs}     # info de cada día
        self.paused, self.clock = False, 0.0
        self.action_buttons[1].label = "Pausa"
        for b in self.agent_buttons:
            b.primary = b.label == self.agent_name and self.mode == "Solo ropa"
        for b in self.mode_buttons:
            b.primary = b.label == self.mode
        self.static = ui._to_display_format(self.render_static())

    def step_day(self) -> None:
        for name, env in self.envs.items():
            self.obs[name], _, _, _, info = env.step(self.policies[name](self.obs[name], env))
            self.history[name].append(info)

    @property
    def day(self) -> int:
        return len(self.history["agent"])

    def cumulative(self, name: str) -> np.ndarray:
        return np.cumsum([d["revenue"] for d in self.history[name]])

    # ---------- eventos ----------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or self.back_button.handle_event(event):
            self.done = True
            return
        if self.bundle is None:
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.toggle_pause()
        for b in self.agent_buttons:
            if b.handle_event(event):
                if self.mode == "Tres productos":
                    self.toast.show("Con tres productos solo hay PPO: DQN no admite acciones MultiDiscrete")
                elif b.label != self.agent_name:
                    self.agent_name = b.label
                    self.restart()
        for b in self.mode_buttons:
            if b.handle_event(event) and b.label != self.mode:
                self.mode = b.label
                self.restart()
        reset, pause, reseed = self.action_buttons
        if reset.handle_event(event):
            self.restart()
        if pause.handle_event(event):
            self.toggle_pause()
        if reseed.handle_event(event):
            self.restart(new_seed=True)
            self.toast.show(f"Otro escenario de clientes (semilla {self.seed})")
        was_dragging = self.price_slider.dragging
        self.price_slider.handle_event(event)
        if was_dragging and not self.price_slider.dragging:   # al soltar: episodio nuevo con ese precio
            self.restart()
        self.speed_slider.handle_event(event)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.action_buttons[1].label = "Seguir" if self.paused else "Pausa"

    def update(self, dt: float) -> None:
        if self.bundle is None or self.paused or self.day >= EPISODE_DAYS or self.price_slider.dragging:
            return
        self.clock += min(dt, 0.1) * self.speed_slider.value   # días que tocan según el tiempo real
        while self.clock >= 1 and self.day < EPISODE_DAYS:
            self.clock -= 1
            self.step_day()

    # ---------- dibujo ----------
    def render_static(self) -> pygame.Surface:
        layer = ui.background(self.size).copy()
        m = px(THEME.margin)
        ui.blit_eyebrow(layer, EYEBROW, (m, px(22)), THEME.accent_text)
        refs = "  ·  ".join(f"{PRODUCT_NAMES[p]} {money(self.cfg.ref_prices[self.cfg.products.index(p)])}"
                            for p in self.products)
        parts = [f"Precio de referencia (precio medio del dataset): {refs}", "1 paso = 1 día",
                 "mismos clientes para las 3 políticas"]
        while len(parts) > 1 and ui.font(THEME.size_label).size("  ·  ".join(parts))[0] > self.size[0] - 2 * m:
            parts.pop()   # lo que no cabe se omite (sigue explicado en la nota y en el informe)
        ui.blit_text(layer, "  ·  ".join(parts), (m, px(94)), THEME.size_label, THEME.text_muted)
        ui.hairline(layer, m, self.size[0] - m, px(120))
        x = px(self.px0)
        ui.vhairline(layer, x - px(24), px(134), px(600))
        ui.blit_eyebrow(layer, "Agente", (x, px(134)))
        ui.blit_eyebrow(layer, "Productos", (x, px(196)))
        ui.blit_eyebrow(layer, "Precio inicial", (x, px(254)))
        ui.blit_eyebrow(layer, "Velocidad", (x, px(312)))
        ui.hairline(layer, x, x + px(self.pw), px(412))
        ui.blit_eyebrow(layer, "Ingreso acumulado", (m, px(134)))
        what = "Precio del agente" if len(self.products) == 1 else "Precio del agente por producto"
        ui.blit_eyebrow(layer, f"{what} (% del precio de referencia)", (m, px(380)))
        # Leyenda de segmentos (franja bajo la gráfica de precio).
        lx, ly = m, self.strip_y + px(30)
        ui.blit_text(layer, "Segmento del día:", (lx, ly), THEME.size_caption, THEME.text_faint)
        lx += px(126)
        for s, seg in enumerate(self.cfg.segments):
            pygame.draw.rect(layer, SEGMENT_COLORS[s], (lx, ly + px(5), px(14), px(6)))
            r = ui.blit_text(layer, SEGMENT_NAMES[seg], (lx + px(20), ly), THEME.size_caption, THEME.text_muted)
            lx = r.right + px(16)
        if len(self.products) == 1:
            ui.dashed_polyline(layer, THEME.text_muted, [(lx, ly + px(9)), (lx + px(18), ly + px(9))], 1, px(3), px(3))
            ui.blit_text(layer, "óptimo del segmento", (lx + px(24), ly), THEME.size_caption, THEME.text_muted)
        return layer

    def draw(self, surface: pygame.Surface) -> None:
        if self.bundle is None:
            self.draw_error(surface)
            return
        surface.blit(self.static, (0, 0))
        self.draw_header(surface)
        self.draw_revenue(surface)
        self.draw_prices(surface)
        self.draw_panel(surface)
        hover = self.hover_day()
        if hover is not None:
            self.draw_tooltip(surface, hover)
        self.back_button.draw(surface)
        self.toast.draw(surface)
        self.update_cursor()
        self.fade.draw(surface)

    def draw_error(self, surface) -> None:
        surface.blit(ui.background(self.size), (0, 0))
        m = px(THEME.margin)
        ui.blit_eyebrow(surface, EYEBROW, (m, px(22)), THEME.accent_text)
        ui.blit_text(surface, "No se pudieron cargar los agentes entrenados", (m, px(50)), THEME.size_headline,
                     weight="serif")
        ui.blit_text(surface, self.error or "", (m, px(104)), THEME.size_label, THEME.text_muted)
        ui.blit_text(surface, "Ejecuta:  python -m task_c_pricing.train", (m, px(132)), THEME.size_body)
        self.back_button.draw(surface)
        self.fade.draw(surface)

    def gain_pct(self) -> float | None:
        if not self.day:
            return None
        fixed = self.cumulative("fixed")[-1]
        return 100 * (self.cumulative("agent")[-1] / fixed - 1) if fixed else None

    def draw_header(self, surface) -> None:
        m = px(THEME.margin)
        gain = self.gain_pct()
        if gain is None:
            text = "¿Vende más un agente de RL que un precio fijo?"
        else:
            sign = "+" if gain >= 0 else "−"
            text = f"{self.model_key}: {sign}{fmt(abs(gain))} % frente al mejor precio fijo"
            if self.day < EPISODE_DAYS:
                text += f" (día {self.day})"
        size = ui.fit_size(text, THEME.size_headline, THEME.size_title, px(self.px0 - 60) - m)
        ui.blit_text(surface, text, (m, px(84)), size, weight="serif", anchor="bottomleft")

    # --- gráfica de ingreso acumulado ---
    def x_of(self, day: float, plot: pygame.Rect) -> float:
        return plot.left + (day - 0.5) / EPISODE_DAYS * plot.width

    def draw_axes_days(self, surface, plot) -> None:
        for d in (1, 10, 20, 30, 40, 50, 60):
            ui.blit_text(surface, str(d), (self.x_of(d, plot), plot.bottom + px(4)), THEME.size_caption,
                         THEME.text_faint, anchor="midtop")

    def draw_revenue(self, surface) -> None:
        plot = self.revenue_plot
        surface.blit(ui.rounded_surface(plot.inflate(px(16), px(16)).size, px(THEME.radius), THEME.chart_bg),
                     plot.inflate(px(16), px(16)))
        curves = {name: self.cumulative(name) for name in self.envs}
        # Escala estable: lo que se ganaría en el episodio completo al ritmo de hoy (no salta en cada día).
        top = 1.05 * max(c[-1] / len(c) * EPISODE_DAYS for c in curves.values()) if self.day else 1e5
        step = 10 ** int(np.floor(np.log10(max(top, 1) / 4)))
        step *= 5 if top / step > 20 else (2 if top / step > 8 else 1)
        for v in np.arange(0, top + 1, step):
            y = plot.bottom - v / top * plot.height
            ui.hairline(surface, plot.left, plot.right, int(y), THEME.grid if v else (*THEME.text, 60))
            ui.blit_text(surface, f"{v / 1000:.0f}k" if v else "0", (plot.right + px(8), y), THEME.size_caption,
                         THEME.text_faint, anchor="midleft")
        self.draw_axes_days(surface, plot)
        if not self.day:
            ui.blit_text(surface, "El episodio empieza en un momento…", plot.center, THEME.size_label,
                         THEME.text_faint, anchor="center")
            return
        ends = {}
        for name in ("oracle", "fixed", "agent"):
            pts = [(plot.left, plot.bottom)] + [(self.x_of(d + 1, plot), plot.bottom - v / top * plot.height)
                                                for d, v in enumerate(curves[name])]
            color = POLICY_COLORS[name]
            if name == "fixed":
                ui.dashed_polyline(surface, color, pts, px(2), px(6), px(4))
            else:
                ui.aa_polyline(surface, color, pts, px(3) if name == "agent" else px(1.5))
            ends[name] = pts[-1]
        # Etiquetas directas al final de cada curva (sin encimarse).
        order = sorted(ends, key=lambda k: ends[k][1])
        last_y = -1e9
        for name in order:
            y = max(ends[name][1], last_y + px(34))
            label = self.model_key if name == "agent" else POLICY_LABELS[name]
            ui.blit_text(surface, label, (self.labels_x, y - px(9)), THEME.size_caption, POLICY_COLORS[name],
                         weight="semibold", anchor="midleft")
            ui.blit_text(surface, money(curves[name][-1]), (self.labels_x, y + px(8)), THEME.size_label,
                         THEME.text if name == "agent" else THEME.text_muted,
                         weight="semibold" if name == "agent" else "regular", anchor="midleft")
            last_y = y

    # --- gráfica de precio ---
    def draw_prices(self, surface) -> None:
        plot = self.price_plot
        surface.blit(ui.rounded_surface(plot.inflate(px(16), px(16)).size, px(THEME.radius), THEME.chart_bg),
                     plot.inflate(px(16), px(16)))
        lo, hi = self.cfg.price_multipliers[0], self.cfg.price_multipliers[-1]
        y_of = lambda mult: plot.bottom - (mult - lo) / (hi - lo) * plot.height
        for mult in (0.5, 1.0, 1.5):
            ui.hairline(surface, plot.left, plot.right, int(y_of(mult)), THEME.grid)
            ui.blit_text(surface, f"{mult:.0%}", (plot.right + px(8), y_of(mult)), THEME.size_caption,
                         THEME.text_faint, anchor="midleft")
        self.draw_axes_days(surface, pygame.Rect(plot.left, self.strip_y, plot.width, px(8)))
        hist = self.history["agent"]
        if len(self.products) == 1:   # mejor precio fijo como referencia
            fixed = self.cfg.price_multipliers[self.best_fixed[0]]
            ui.dashed_polyline(surface, POLICY_COLORS["fixed"], [(plot.left, y_of(fixed)), (plot.right, y_of(fixed))],
                               1, px(5), px(4))
        if not hist:
            return
        slot = plot.width / EPISODE_DAYS
        for d, info in enumerate(hist):   # franja del segmento de cada día
            pygame.draw.rect(surface, SEGMENT_COLORS[info["segment"]],
                             (int(plot.left + d * slot), self.strip_y, max(1, int(slot) - 1), px(6)))
        refs = np.array(self.cfg.ref_prices)
        label_ys = []
        for p in range(len(self.products)):
            ref_index = self.cfg.products.index(self.products[p])
            if len(self.products) == 1:   # óptimo del segmento (lo que haría el oráculo sin límite de movimiento)
                opt = [(self.x_of(d + 1, plot) + dx, y_of(self.cfg.price_multipliers[
                    self.cfg.optimal_level(info["segment"], ref_index)]))
                       for d, info in enumerate(hist) for dx in (-slot / 2, slot / 2)]
                ui.dashed_polyline(surface, THEME.text_muted, opt, 1, px(3), px(3))
            pts = []
            for d, info in enumerate(hist):
                y = y_of(info["prices"][p] / refs[ref_index])
                x = self.x_of(d + 1, plot)
                pts += [(x - slot / 2, y), (x + slot / 2, y)]
            ui.aa_polyline(surface, PRODUCT_COLORS[p], pts, px(2.5))
            label_ys.append((pts[-1][1], p))
        if len(self.products) > 1:   # etiquetas directas al final de cada línea, sin encimarse
            last = -1e9
            for y, p in sorted(label_ys):
                y = max(y, last + px(18))
                ui.blit_text(surface, PRODUCT_NAMES[self.products[p]], (self.labels_x, y), THEME.size_caption,
                             PRODUCT_COLORS[p], weight="semibold", anchor="midleft")
                last = y

    # --- panel ---
    def draw_panel(self, surface) -> None:
        x, right = px(self.px0), px(self.px0 + self.pw)
        for b in (*self.agent_buttons, *self.mode_buttons, *self.action_buttons):
            b.draw(surface)
        if self.mode == "Tres productos":   # DQN no aplica con varios productos
            r = self.agent_buttons[1].rect
            pygame.draw.line(surface, THEME.text_faint, (r.left + px(4), r.centery), (r.right - px(4), r.centery))
        mult = self.cfg.price_multipliers[int(round(self.price_slider.value))]
        value = money(mult * self.cfg.ref_prices[0]) if len(self.products) == 1 else f"{mult:.0%}"
        ui.blit_text(surface, value, (right, px(250)), THEME.size_title, weight="serif", anchor="topright")
        ui.blit_text(surface, f"{fmt(self.speed_slider.value, 0)} días/s", (right, px(308)), THEME.size_title,
                     weight="serif", anchor="topright")
        self.price_slider.draw(surface, stops=self.cfg.n_levels)
        self.speed_slider.draw(surface)

        rows = [("Día", f"{self.day} de {EPISODE_DAYS}" + ("  ·  terminado" if self.day >= EPISODE_DAYS else ""))]
        if self.day:
            today = self.history["agent"][-1]
            seg = today["segment"]
            prices = today["prices"]
            price_txt = money(prices[0]) if len(prices) == 1 else " · ".join(money(p) for p in prices)
            rows += [("Segmento", SEGMENT_NAMES[self.cfg.segments[seg]]),
                     ("Precio hoy", price_txt),
                     (f"Ingreso {self.model_key}", money(self.cumulative("agent")[-1])),
                     ("Mejor precio fijo", money(self.cumulative("fixed")[-1])),
                     ("Oráculo (techo)", money(self.cumulative("oracle")[-1]))]
        y = px(422)
        for label, value in rows:
            ui.blit_text(surface, label, (x, y), THEME.size_label, THEME.text_faint)
            ui.blit_text(surface, value, (right, y), THEME.size_label, THEME.text, weight="semibold",
                         anchor="topright")
            y += px(21)
        note = self.evaluation_note()
        if note:
            y = px(556)
            for line in ui.wrap_text(note, THEME.size_caption, px(self.pw)):
                ui.blit_text(surface, line, (x, y), THEME.size_caption, THEME.text_muted)
                y += px(18)

    def evaluation_note(self) -> str:
        """Resultado de la evaluación completa (200 episodios), para no sacar conclusiones de un solo episodio."""
        summary = self.bundle["summary"]
        if not summary:
            return ""
        block = summary["multi" if self.mode == "Tres productos" else "single"]
        stats = block.get("PPO" if self.mode == "Tres productos" else self.agent_name)
        if not stats:
            return ""
        return (f"En {summary['eval_episodes']} episodios: {fmt(stats['gain_pct'])} % sobre el mejor precio fijo; "
                f"gana en el {fmt(stats['win_rate_pct'], 0)} % de los episodios.")

    def hover_day(self):
        x, y = self.mouse
        plots = (self.revenue_plot, self.price_plot)
        if not self.day or not any(p.left <= x <= p.right and p.top - px(8) <= y <= p.bottom + px(8) for p in plots):
            return None
        d = int((x - self.revenue_plot.left) / self.revenue_plot.width * EPISODE_DAYS)
        return d if 0 <= d < self.day else None

    def draw_tooltip(self, surface, d: int) -> None:
        for plot in (self.revenue_plot, self.price_plot):
            ui.vhairline(surface, int(self.x_of(d + 1, plot)), plot.top, plot.bottom, (*THEME.text, 80))
        info = self.history["agent"][d]
        title = ui.render_text(f"Día {d + 1} · {SEGMENT_NAMES[self.cfg.segments[info['segment']]]}",
                               THEME.size_label, THEME.text, weight="semibold")
        prices = " · ".join(money(p) for p in info["prices"])
        rows = [("Precio del agente", prices, THEME.accent),
                (f"Ingreso del día ({self.model_key})", money(info["revenue"]), THEME.accent),
                ("Ingreso del día (fijo)", money(self.history["fixed"][d]["revenue"]), POLICY_COLORS["fixed"]),
                ("Ingreso del día (oráculo)", money(self.history["oracle"][d]["revenue"]), POLICY_COLORS["oracle"]),
                ("Clientes interesados", " · ".join(str(n) for n in info["shoppers"]), THEME.text_muted)]
        box = pygame.Rect(0, 0, max(title.get_width(), px(290)) + px(28), px(40) + px(21) * len(rows))
        box.topleft = (self.mouse[0] + px(16), self.mouse[1] - box.height // 2)
        if box.right > px(self.px0 - 24):
            box.right = self.mouse[0] - px(16)
        box.clamp_ip(surface.get_rect().inflate(-px(8), -px(8)))
        solid_box(surface, box)
        surface.blit(title, (box.left + px(14), box.top + px(10)))
        y = box.top + px(36)
        for label, value, color in rows:
            ui.aa_line(surface, color, (box.left + px(14), y + px(9)), (box.left + px(24), y + px(9)), px(2))
            ui.blit_text(surface, label, (box.left + px(32), y), THEME.size_label, THEME.text_muted)
            ui.blit_text(surface, value, (box.right - px(14), y), THEME.size_label, THEME.text, weight="semibold",
                         anchor="topright")
            y += px(21)

    def update_cursor(self) -> None:
        buttons = (*self.agent_buttons, *self.mode_buttons, *self.action_buttons, self.back_button)
        hand = any(b.rect.collidepoint(self.mouse) for b in buttons) or self.price_slider.dragging \
            or self.speed_slider.dragging
        if hand != self._cursor:
            self._cursor = hand
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hand else pygame.SYSTEM_CURSOR_ARROW)
            except pygame.error:  # sin ventana real (p. ej. pruebas headless)
                pass


def run(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    """Contrato con el menú: bucle propio; vuelve con ESC / "← Menú"."""
    try:
        view = PricingScreen(load_bundle(), size=screen.get_size())
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        view = PricingScreen(None, str(exc), size=screen.get_size())
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
