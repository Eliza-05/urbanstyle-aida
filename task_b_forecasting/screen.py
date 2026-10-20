"""Tarea B (Elizabeth): pantalla Pygame del pronóstico de demanda, en lenguaje editorial (periodismo de datos).

Lectura en pocos segundos: un titular con la conclusión de la serie, la gráfica como protagonista
(barras de ventas reales, LSTM con su banda, líneas base y etiquetas directas), un marcador tipográfico
con el error en prueba y una tira de 6 mini-gráficas para elegir la serie (con el slider debajo).
Solo CARGA el checkpoint ya entrenado (models/lstm_forecast.pt) y los archivos de outputs/task_b/; nunca entrena.
Toda la geometría se escribe en píxeles lógicos (ventana base 1024×640) y se convierte con ui.px().
"""
import math

import numpy as np
import pandas as pd
import pygame

from common import config
from common import ui
from common.ui import THEME, px

COLORS = THEME.series_colors
ANIM_MS = 250                       # duración de la transición al cambiar de serie
LABELS = {                          # etiquetas directas (y clave de visibilidad) de cada elemento
    "actual": "Ventas reales",
    "lstm": "LSTM",
    "moving_average": "Media móvil",
    "hist_mean": "Media histórica",
}
MODEL_ROWS = (  # (nombre visible, columna en metrics.csv, clave de color)
    ("LSTM", "lstm", "lstm"),
    ("Media móvil", "moving_average", "moving_average"),
    ("Media histórica", "hist_mean_trainval", "hist_mean"),
    ("Ingenua", "naive", "naive"),
)
MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
PART_NAMES = {"train": "entrenamiento", "val": "validación", "test": "prueba"}

_cache = {}  # datos cargados una sola vez por sesión


# ---------------------------------------------------------------------------
# Datos: todo sale del checkpoint y de outputs/task_b/ (no se entrena ni se recalculan métricas)
# ---------------------------------------------------------------------------

def load_cached():
    """Carga (una vez) checkpoint, predicciones y métricas. Import diferido: torch solo se carga al entrar."""
    if "results" not in _cache:
        from task_b_forecasting.evaluate import load_results
        results = load_results()
        results["view"] = build_view_data(results)
        _cache["results"] = results
    return _cache["results"]


def build_view_data(results) -> dict:
    """Arreglos por serie listos para dibujar (semanas 0..51) + textos derivados de los archivos guardados."""
    from task_b_forecasting.prepare import scaler_from_minmax
    from task_b_forecasting.train import predict_units

    ckpt, weekly, pred = results["ckpt"], results["weekly"], results["predictions"]
    names = ckpt["series_names"]
    n_weeks, window = len(weekly), ckpt["window"]

    # LSTM desde la primera semana con `window` semanas previas (semana 5), con el MISMO modelo guardado.
    scaler = scaler_from_minmax(ckpt["scaler_min"], ckpt["scaler_max"])
    lstm_pred, s_idx, t_idx = predict_units(results["models"], scaler.transform(weekly.to_numpy(float)),
                                            window, range(window, n_weeks), scaler)
    lstm = np.full((n_weeks, len(names)), np.nan)
    lstm[t_idx, s_idx] = lstm_pred

    def column(col):
        table = pred.pivot(index="week_start", columns="series", values=col)[names]
        out = np.full((n_weeks, len(names)), np.nan)
        out[n_weeks - len(table):] = table.to_numpy(float)
        return out

    val = pred[pred["part"] == "val"]
    val_mae = (val["lstm"] - val["actual"]).abs().groupby(val["series"]).mean()[names].to_numpy()
    test = pred[pred["part"] == "test"]
    inside = ((test["lstm"] - test["actual"]).abs() <= test["series"].map(dict(zip(names, val_mae))))
    inside_by_series = inside.groupby(test["series"]).sum()[names].to_numpy()

    nxt = results["next_week"].set_index("series").loc[names]
    start = pd.Timestamp(nxt["week_start"].iloc[0])
    return {
        "names": names,
        "actual": weekly[names].to_numpy(float),
        "lstm": lstm,
        "moving_average": column("moving_average"),
        "hist_mean": column("hist_mean_trainval"),
        "val_mae": val_mae,
        "inside": inside_by_series,
        "inside_total": int(inside.sum()),
        "test_per_series": int(len(test) / len(names)),
        "next": nxt[["lstm", "moving_average", "hist_mean_trainval"]].to_numpy(float),
        "next_start": start,
        "next_end": start + pd.Timedelta(6, unit="D"),
        "first_lstm_week": window,
        "first_ma_week": ckpt["ma_k"],
        "significance_note": significance_note(),
    }


def significance_note() -> str:
    """Nota de significancia redactada a partir de outputs/task_b/metrics/bootstrap.csv (si existe)."""
    from task_b_forecasting.paths import BOOTSTRAP_CSV
    if not BOOTSTRAP_CSV.exists():
        return ""
    boot = pd.read_csv(BOOTSTRAP_CSV).set_index("model")
    contains_zero = lambda m: boot.loc[m, "dmae_lo"] <= 0 <= boot.loc[m, "dmae_hi"] and \
        boot.loc[m, "drmse_lo"] <= 0 <= boot.loc[m, "drmse_hi"]
    if all(contains_zero(m) for m in ("moving_average", "hist_mean_train", "hist_mean_trainval")):
        return "LSTM, media móvil y media histórica: sin diferencia significativa (bootstrap 95 %)."
    return "Ver significancia en outputs/task_b/metrics/bootstrap.csv."


def headline(name: str, value: float) -> str:
    """Titular con la conclusión de la serie, a partir del pronóstico real de la LSTM (redondeado)."""
    units = int(round(value))
    if units == 1:
        return f"{name}: se espera una unidad la próxima semana"
    return f"{name}: se esperan unas {units} unidades la próxima semana"


def nice_ceiling(value: float) -> float:
    """Máximo del eje Y redondeado hacia arriba a un múltiplo de 5."""
    return max(10.0, math.ceil(value / 5.0) * 5.0)


def fmt(value: float, decimals: int = 1) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{decimals}f}".replace(".", ",")


def fmt_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """'24–30 dic' o '31 dic – 6 ene'."""
    if start.month == end.month:
        return f"{start.day}–{end.day} {MONTHS[end.month - 1]}"
    return f"{start.day} {MONTHS[start.month - 1]} – {end.day} {MONTHS[end.month - 1]}"


def spread_labels(items, gap: int, low: int, high: int) -> dict:
    """Reparte etiquetas en vertical para que no se encimen: items = [(clave, y_deseada, alto)]."""
    items = sorted(items, key=lambda it: it[1])
    heights = [h for _, _, h in items]
    ys = [y for _, y, _ in items]
    # Pasada hacia abajo: cada etiqueta, al menos `gap` por debajo de la anterior (y dentro del límite superior).
    for i in range(len(ys)):
        floor = low + heights[i] / 2 if i == 0 else ys[i - 1] + (heights[i - 1] + heights[i]) / 2 + gap
        ys[i] = max(ys[i], floor)
    # Pasada hacia arriba: si alguna se sale por abajo, se sube el grupo lo necesario.
    for i in range(len(ys) - 1, -1, -1):
        ceiling = high - heights[i] / 2 if i == len(ys) - 1 else ys[i + 1] - (heights[i + 1] + heights[i]) / 2 - gap
        ys[i] = min(ys[i], ceiling)
    return {key: y for (key, _, _), y in zip(items, ys)}


def solid_box(surface, box: pygame.Rect) -> None:
    """Fondo de tooltip: sólido y opaco (sin sombra), con un filete fino."""
    pygame.draw.rect(surface, THEME.bg_raised, box, border_radius=px(THEME.radius))
    surface.blit(ui.rounded_surface(box.size, px(THEME.radius), (*THEME.bg_raised, 0), THEME.hairline_color), box)


def R(x, y, w, h) -> pygame.Rect:
    """Rect en píxeles físicos a partir de coordenadas lógicas."""
    return pygame.Rect(px(x), px(y), px(w), px(h))


# ---------------------------------------------------------------------------
# Pantalla
# ---------------------------------------------------------------------------

class ForecastScreen:
    """Estado e interacción de la pantalla. Métodos: handle_event, update(dt), draw(surface)."""

    def __init__(self, results=None, error: str | None = None, size=None):
        self.size = size or (px(config.WINDOW_WIDTH), px(config.WINDOW_HEIGHT))
        self.results = results
        self.error = error
        self.done = False
        self.series_idx = 0
        self.visible = {key: True for key in LABELS}
        self.mouse = (-1, -1)
        self.toast = ui.Toast()
        self.fade = ui.FadeIn()
        self.legend_rects, self.name_rects = {}, []   # zonas clicables (etiquetas directas y mini-gráficas)
        self.info_rect = pygame.Rect(0, 0, 0, 0)       # "cómo leer la gráfica"
        self.scope = "series"                          # marcador: "series" o "global"
        self._cursor = None
        self.layout()
        self.back_button = ui.Button(self.back_rect, "← Menú")
        if results is None:
            return
        self.view = results["view"]
        self.shown = self.target_vector(0)
        self.anim_from = self.anim_to = self.shown
        self.anim_start = -ANIM_MS
        self.slider = ui.Slider(self.slider_rect, 0, len(self.view["names"]) - 1, 0)
        self.knob_target = 0.0
        self.export_button = ui.Button(self.export_rect, "Exportar CSV", primary=True)
        self.band_layer = ui._to_display_format(pygame.Surface(self.plot.size, pygame.SRCALPHA))
        self.band_gradient = ui.gradient_surface(self.plot.size, COLORS["lstm"], COLORS["lstm"],
                                                 vertical=True, alpha=(34, 8))
        n = len(self.view["actual"])
        self.xs = self.plot_weeks.left + (np.arange(n) + 0.5) * self.plot_weeks.width / n  # centro de cada semana
        self.sparks = {}                               # mini-gráficas pre-renderizadas
        self.static = ui._to_display_format(self.render_static())
        # Búfer con lo que solo cambia con el estado: se redibuja cuando cambia su clave (60 FPS).
        self.buffer = self.static.copy()
        self._side_key = self._chart_key = None

    # ---------- geometría (en px lógicos -> físicos) ----------
    def layout(self) -> None:
        W, H = self.size[0] / THEME.scale, self.size[1] / THEME.scale   # tamaño lógico
        m = THEME.margin
        self.W, self.H = W, H
        # Enlaces de texto arriba a la derecha.
        back_w = ui.font(THEME.size_body, weight="medium").size("← Menú")[0] / THEME.scale + 12
        export_w = ui.font(THEME.size_body, weight="medium").size("Exportar CSV")[0] / THEME.scale + 12
        self.back_rect = R(W - m - back_w, 14, back_w, 30)
        self.export_rect = R(W - m - back_w - 26 - export_w, 14, export_w, 30)
        # Marcador lateral y gráfica protagonista.
        score_w = 236
        self.score = R(W - m - score_w, 150, score_w, 340)
        chart_right = W - m - score_w - 32
        self.plot = R(m, 182, chart_right - m, 304)                 # incluye la columna de etiquetas
        self.y_axis_x = px(chart_right)                            # números del eje Y, alineados a la derecha
        weeks_w = (chart_right - m) - 214
        self.plot_weeks = R(m, 182, weeks_w, 304)                   # semanas 1..52
        self.next_x = px(m + weeks_w + 22)                         # punto de la próxima semana
        self.labels_x = px(m + weeks_w + 42)                       # inicio de las etiquetas directas
        self.grid_right = px(chart_right - 24)
        # Tira de mini-gráficas (selector de serie) y slider alineado con ellas.
        gap = 24
        item_w = (W - 2 * m - 5 * gap) / 6
        self.strip = [R(m + i * (item_w + gap), 544, item_w, 58) for i in range(6)]
        first, last = self.strip[0].centerx, self.strip[-1].centerx
        self.slider_rect = pygame.Rect(first, px(622) - px(4), last - first, px(8))
        self.segment_rects = {}                                    # se fijan al dibujar el marcador

    # ---------- datos animados ----------
    def target_vector(self, s: int) -> np.ndarray:
        """Todo lo que cambia con la serie, en un solo vector para interpolarlo de una vez."""
        v = self.view
        band = v["val_mae"][s]
        peak = np.nanmax(np.concatenate([v["actual"][:, s], v["lstm"][:, s] + band, v["next"][s] + band]))
        return np.concatenate([v["actual"][:, s], v["lstm"][:, s], v["moving_average"][:, s],
                               v["hist_mean"][:, s], v["next"][s], [band, nice_ceiling(peak)]])

    def unpack(self, vec: np.ndarray) -> dict:
        n = len(self.view["actual"])
        return {"actual": vec[:n], "lstm": vec[n:2 * n], "moving_average": vec[2 * n:3 * n],
                "hist_mean": vec[3 * n:4 * n], "next": vec[4 * n:4 * n + 3], "band": vec[-2], "ymax": vec[-1]}

    def set_series(self, idx: int, snap_knob: bool = True) -> None:
        idx = max(0, min(len(self.view["names"]) - 1, idx))
        if snap_knob:
            self.knob_target = float(idx)
        if idx == self.series_idx:
            return
        self.series_idx = idx
        self.anim_from = self.shown.copy()
        self.anim_to = self.target_vector(idx)
        self.anim_start = pygame.time.get_ticks()

    @property
    def series_name(self) -> str:
        return self.view["names"][self.series_idx]

    # ---------- eventos ----------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or self.back_button.handle_event(event):
            self.done = True
            return
        if self.results is None:
            return
        if self.export_button.handle_event(event):
            from task_b_forecasting.evaluate import export_all
            paths = export_all(self.results)
            try:
                folder = paths[0].parent.relative_to(config.ROOT).as_posix()  # ruta corta y relativa al repo
            except ValueError:
                folder = str(paths[0].parent)
            self.toast.show(f"Exportado: {' y '.join(path.name for path in paths)} en {folder}/")
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
            self.set_series(self.series_idx + (1 if event.key == pygame.K_RIGHT else -1))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.click_targets(event.pos):
            pass
        else:
            was_dragging = self.slider.dragging
            value = self.slider.handle_event(event)
            if self.slider.dragging:          # el control sigue al mouse; la serie cambia al cruzar cada marca
                self.set_series(round(value), snap_knob=False)
            elif was_dragging:                # al soltar, encaja en la serie más cercana
                self.set_series(round(value))

    def click_targets(self, pos) -> bool:
        """Clic en una etiqueta directa (mostrar/ocultar), una mini-gráfica o el selector. True si lo atendió."""
        for key, rect in self.legend_rects.items():
            if rect.collidepoint(pos):
                self.visible[key] = not self.visible[key]
                return True
        for i, rect in enumerate(self.name_rects):
            if rect.collidepoint(pos):
                self.set_series(i)
                return True
        for scope, rect in self.segment_rects.items():
            if rect.collidepoint(pos):
                self.scope = scope
                return True
        return False

    # ---------- actualización por cuadro ----------
    def update(self, dt: float) -> None:
        if self.results is None:
            return
        t = min(1.0, (pygame.time.get_ticks() - self.anim_start) / ANIM_MS)
        ease = 1 - (1 - t) ** 3  # ease-out: arranca rápido y frena suave
        both = np.isfinite(self.anim_from) & np.isfinite(self.anim_to)
        self.shown = np.where(both, self.anim_from + (self.anim_to - self.anim_from) * ease, self.anim_to)
        if not self.slider.dragging:  # el control "encaja" suavemente en la marca de la serie
            self.slider.value += (self.knob_target - self.slider.value) * min(1.0, dt * 18)

    # ---------- dibujo ----------
    def render_static(self) -> pygame.Surface:
        """Lo que no cambia nunca se dibuja una sola vez: fondo, etiqueta de sección y línea de contexto."""
        layer = ui.background(self.size).copy()
        m = px(THEME.margin)
        ui.blit_eyebrow(layer, "Tarea B · Pronóstico de demanda", (m, px(22)), THEME.accent_text)
        context = ui.blit_text(layer, "3 categorías originales × género = 6 series  ·  unidades por semana, 2023",
                               (m, px(96)), THEME.size_label, THEME.text_muted)
        # "Cómo leer la gráfica": ayuda emergente con las notas (antes eran texto fijo).
        x = context.right + px(18)
        ui.aa_circle(layer, THEME.accent_text, (x + px(7), context.centery), px(7))
        ui.blit_text(layer, "i", (x + px(7), context.centery), THEME.size_caption, THEME.bg, weight="bold",
                     anchor="center")
        link = ui.blit_text(layer, "cómo leer la gráfica", (x + px(20), context.centery), THEME.size_label,
                            THEME.accent_text, anchor="midleft")
        self.info_rect = pygame.Rect(x, context.top, link.right - x, context.height)
        ui.hairline(layer, m, self.size[0] - m, px(512))   # separa la gráfica de la tira de series
        # Rótulo de la tira: explica qué son los números de cada serie.
        ui.blit_eyebrow(layer, "Pronóstico próxima semana (LSTM)", (m, px(520)))
        return layer

    def draw(self, surface: pygame.Surface) -> None:
        if self.results is None:
            self.draw_error(surface)
            return
        self.refresh_buffer()
        surface.blit(self.buffer, (0, 0))
        hover = self.hover_week()
        if hover is not None:
            self.draw_tooltip(surface, hover)
        self.slider.draw(surface, stops=len(self.view["names"]))
        if self.info_rect.collidepoint(self.mouse):
            self.draw_info_tooltip(surface)
        self.export_button.draw(surface)
        self.back_button.draw(surface)
        self.toast.draw(surface)
        self.update_cursor()
        self.fade.draw(surface)

    def refresh_buffer(self) -> None:
        """Redibuja en el búfer solo las zonas cuyo estado cambió (60 FPS sin recalcular todo)."""
        seg_hover = next((k for k, r in self.segment_rects.items() if r.collidepoint(self.mouse)), None)
        strip_hover = next((i for i, r in enumerate(self.name_rects) if r.collidepoint(self.mouse)), None)
        side_key = (self.series_idx, self.scope, seg_hover, strip_hover)
        if self._side_key != side_key:
            for rect in self.side_areas():
                self.buffer.blit(self.static, rect, rect)
            self.draw_header(self.buffer)
            self.draw_scoreboard(self.buffer)
            self.draw_strip(self.buffer)
            self._side_key = side_key
        animating = pygame.time.get_ticks() - self.anim_start < ANIM_MS + 20
        legend_hover = next((k for k, r in self.legend_rects.items() if r.collidepoint(self.mouse)), None)
        key = (self.series_idx, tuple(self.visible.values()), self.hover_week(), legend_hover)
        if animating or key != self._chart_key:
            area = self.chart_area()
            self.buffer.blit(self.static, area, area)
            self.draw_chart(self.buffer, self.unpack(self.shown))
            self._chart_key = None if animating else key

    def side_areas(self):
        m = px(THEME.margin)
        return (pygame.Rect(0, px(40), self.size[0], px(52)),                 # titular
                pygame.Rect(0, px(112), self.size[0], px(24)),                # datos junto al titular
                self.score.inflate(px(8), px(30)),                           # marcador
                pygame.Rect(m - px(4), self.strip[0].top - px(6), self.size[0] - 2 * m + px(8), px(70)))

    def chart_area(self) -> pygame.Rect:
        return pygame.Rect(0, px(140), self.score.left - px(10), px(372))

    def draw_error(self, surface) -> None:
        surface.blit(ui.background(self.size), (0, 0))
        m = px(THEME.margin)
        ui.blit_eyebrow(surface, "Tarea B · Pronóstico de demanda", (m, px(22)), THEME.accent_text)
        ui.blit_text(surface, "No se pudo cargar el modelo entrenado", (m, px(50)), THEME.size_headline,
                     weight="serif")
        ui.blit_text(surface, self.error or "", (m, px(104)), THEME.size_label, THEME.text_muted)
        ui.blit_text(surface, "Ejecuta:  python -m task_b_forecasting.train", (m, px(132)), THEME.size_body)
        self.back_button.draw(surface)
        self.fade.draw(surface)

    # --- cabecera: titular con la conclusión + datos pequeños ---
    def draw_header(self, surface) -> None:
        v, s = self.view, self.series_idx
        m = px(THEME.margin)
        text = headline(self.series_name, v["next"][s][0])
        width = self.size[0] - 2 * m
        size = ui.fit_size(text, THEME.size_headline, THEME.size_title, width)
        ui.blit_text(surface, text, (m, px(86)), size, THEME.text, weight="serif", anchor="bottomleft")
        # Datos pequeños: ventas de la última semana y MAE de la LSTM en prueba.
        block = self.results["metrics"].set_index("series").loc[self.series_name]
        last_start = self.results["weekly"].index[-1]
        x, y = m, px(116)
        for label, value in (("Última semana", f"{fmt(v['actual'][-1, s], 0)} u "
                                                f"({fmt_range(last_start, last_start + pd.Timedelta(6, unit='D'))})"),
                             ("MAE de la LSTM en prueba", f"{fmt(block['lstm_mae'], 2)} u/semana")):
            r = ui.blit_text(surface, label, (x, y), THEME.size_label, THEME.text_faint)
            r = ui.blit_text(surface, value, (r.right + px(8), y), THEME.size_label, THEME.text, weight="semibold")
            x = r.right + px(28)

    # --- gráfica protagonista ---
    def x_center(self, week: int) -> float:
        return float(self.xs[week])

    def y_of(self, units: float, ymax: float) -> float:
        return self.plot.bottom - units / ymax * self.plot.height

    def hover_week(self):
        """Semana bajo el mouse: 0..51, "next" en la columna de la próxima semana, o None."""
        x, y = self.mouse
        if not self.plot.top <= y <= self.plot.bottom:
            return None
        if abs(x - self.next_x) <= px(14):
            return "next"
        if self.plot_weeks.left <= x < self.plot_weeks.right:
            n = len(self.view["actual"])
            return int((x - self.plot_weeks.left) / self.plot_weeks.width * n)
        return None

    def draw_chart(self, surface, d) -> None:
        v = self.view
        n = len(v["actual"])
        ymax = d["ymax"]
        hover = self.hover_week()
        self.legend_rects = {}

        # Zona de prueba: único fondo sutil, con su anotación (línea fina + texto).
        val_end = self.results["ckpt"]["val_end"]
        x_test = int(self.plot_weeks.left + val_end * self.plot_weeks.width / n)
        zone = pygame.Rect(x_test, self.plot.top, self.plot_weeks.right - x_test, self.plot.height)
        surface.blit(ui.rounded_surface(zone.size, 0, THEME.chart_bg), zone)
        bracket_y = self.plot.top - px(8)
        ui.hairline(surface, zone.left, zone.right, bracket_y, (*THEME.text, 90))
        for x in (zone.left, zone.right - max(1, px(1))):
            ui.vhairline(surface, x, bracket_y, bracket_y + px(5), (*THEME.text, 90))
        ui.blit_text(surface, f"prueba · {n - val_end} semanas", (zone.centerx, bracket_y - px(4)),
                     THEME.size_caption, THEME.text_muted, anchor="midbottom")

        # Cuadrícula mínima y eje Y a la derecha.
        step = 5 if ymax <= 30 else 10
        for u in range(0, int(ymax) + 1, step):
            y = int(self.y_of(u, ymax))
            ui.hairline(surface, self.plot.left, self.grid_right, y, THEME.grid if u else (*THEME.text, 60))
            ui.blit_text(surface, str(u), (self.y_axis_x, y), THEME.size_caption, THEME.text_faint,
                         anchor="midright")
        weeks_idx = self.results["weekly"].index
        for w, date in enumerate(weeks_idx):
            if w == 0 or date.month != weeks_idx[w - 1].month:
                ui.blit_text(surface, MONTHS[date.month - 1], (self.x_center(w), self.plot.bottom + px(6)),
                             THEME.size_caption, THEME.text_faint, anchor="midtop")

        # Barras de ventas reales: finas y tenues.
        slot = self.plot_weeks.width / n
        if self.visible["actual"]:
            bar_w = max(2, int(slot * 0.42))
            for w in range(n):
                top = self.y_of(d["actual"][w], ymax)
                color = (90, 84, 79) if hover == w else COLORS["actual"]
                pygame.draw.rect(surface, color, (int(self.x_center(w) - bar_w / 2), int(top), bar_w,
                                                  int(self.plot.bottom - top)))
        key = self.direct_label(surface, "actual", (self.plot.left, self.plot.top - px(30)), "midleft")
        self.legend_rects["actual"] = key

        # Banda ± MAE de validación muy sutil y líneas de los modelos (antialias).
        weeks = np.arange(n)
        ok = np.isfinite(d["lstm"])
        lstm_n, ma_n, hist_n = d["next"]
        if self.visible["lstm"]:
            band = d["band"]
            upper = [(self.x_center(w), self.y_of(min(ymax, d["lstm"][w] + band), ymax)) for w in weeks[ok]]
            lower = [(self.x_center(w), self.y_of(max(0, d["lstm"][w] - band), ymax)) for w in weeks[ok][::-1]]
            origin = self.plot.topleft
            self.band_layer.fill((0, 0, 0, 0))
            pygame.draw.polygon(self.band_layer, (255, 255, 255, 255),
                                [(x - origin[0], y - origin[1]) for x, y in upper + lower])
            self.band_layer.blit(self.band_gradient, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            surface.blit(self.band_layer, origin)
        if self.visible["moving_average"]:
            okm = np.isfinite(d["moving_average"])
            pts = [(self.x_center(w), self.y_of(d["moving_average"][w], ymax)) for w in weeks[okm]]
            ui.dashed_polyline(surface, COLORS["moving_average"], pts + [(self.next_x, self.y_of(ma_n, ymax))],
                               px(2), px(6), px(5))
        if self.visible["hist_mean"]:
            okh = np.isfinite(d["hist_mean"])
            if okh.any():
                y = self.y_of(d["hist_mean"][okh][0], ymax)
                ui.aa_line(surface, COLORS["hist_mean"], (self.x_center(weeks[okh][0]) - slot / 2, y),
                           (self.next_x, self.y_of(hist_n, ymax)), px(2))
        if self.visible["lstm"]:
            pts = [(self.x_center(w), self.y_of(d["lstm"][w], ymax)) for w in weeks[ok]]
            ui.aa_polyline(surface, COLORS["lstm"], pts, px(3))
            ui.dashed_polyline(surface, COLORS["lstm"], [pts[-1], (self.next_x, self.y_of(lstm_n, ymax))],
                               px(2), px(4), px(3))

        self.draw_next_week(surface, d, ymax, hover == "next")

    def draw_next_week(self, surface, d, ymax, hovered: bool) -> None:
        """Punto de la próxima semana de cada modelo con su ETIQUETA DIRECTA (clic para mostrar u ocultar)."""
        v = self.view
        lstm_n, ma_n, hist_n = d["next"]
        x = self.next_x
        ui.vhairline(surface, x, self.plot.top, self.plot.bottom, (*THEME.text, 70 if hovered else 40))
        band = d["band"]
        if self.visible["lstm"]:  # barra de error ± MAE
            y_hi, y_lo = self.y_of(min(ymax, lstm_n + band), ymax), self.y_of(max(0, lstm_n - band), ymax)
            soft = tuple(int(c * 0.55 + b * 0.45) for c, b in zip(COLORS["lstm"], THEME.bg))
            ui.aa_line(surface, soft, (x, y_hi), (x, y_lo), px(2))
            for yy in (y_hi, y_lo):
                ui.aa_line(surface, soft, (x - px(5), yy), (x + px(5), yy), px(2))
        points = {"lstm": self.y_of(lstm_n, ymax), "moving_average": self.y_of(ma_n, ymax),
                  "hist_mean": self.y_of(hist_n, ymax)}
        heights = {"lstm": px(56), "moving_average": px(20), "hist_mean": px(20)}
        placed = spread_labels([(k, points[k], heights[k]) for k in points], px(8),
                               self.plot.top, self.plot.bottom)
        for key in ("hist_mean", "moving_average", "lstm"):
            color = COLORS[key] if self.visible[key] else THEME.text_faint
            if self.visible[key]:
                ui.aa_circle(surface, color, (x, points[key]), px(5 if key == "lstm" else 3.5))
            y = placed[key]
            if abs(y - points[key]) > px(3):  # línea guía si la etiqueta se desplazó para no encimarse
                soft = tuple(int(c * 0.5 + b * 0.5) for c, b in zip(color, THEME.bg))
                ui.aa_line(surface, soft, (x + px(7), points[key]), (self.labels_x - px(5), y), 1)
            if key == "lstm":
                self.legend_rects[key] = self.draw_lstm_label(surface, (self.labels_x, y), lstm_n, v, color)
            else:
                value = ma_n if key == "moving_average" else hist_n
                self.legend_rects[key] = self.direct_label(surface, key, (self.labels_x, y), "midleft",
                                                           f"  {fmt(value)}")

    def draw_lstm_label(self, surface, pos, value, v, color) -> pygame.Rect:
        """Etiqueta protagonista: valor grande en serif, ± MAE al lado y nombre con la semana debajo."""
        x, y = pos
        on = self.visible["lstm"]
        num = ui.blit_text(surface, fmt(value), (x, y + px(6)), THEME.size_number,
                           THEME.text if on else THEME.text_faint, weight="serif", anchor="bottomleft")
        ui.blit_text(surface, f"± {fmt(v['val_mae'][self.series_idx])}", (num.right + px(6), num.bottom - px(6)),
                     THEME.size_label, THEME.text_muted, anchor="bottomleft")
        name = ui.blit_text(surface, f"LSTM · {fmt_range(v['next_start'], v['next_end'])}",
                            (x, num.bottom + px(2)), THEME.size_caption, color, weight="semibold")
        if not on:
            pygame.draw.line(surface, THEME.text_faint, (name.left, name.centery), (name.right, name.centery))
        rect = num.union(name).inflate(px(8), px(4))
        if rect.collidepoint(self.mouse):
            ui.hairline(surface, rect.left + px(4), rect.right - px(4), rect.bottom, (*THEME.text, 90))
        return rect

    def direct_label(self, surface, key, pos, anchor, suffix: str = "") -> pygame.Rect:
        """Etiqueta de texto en el color de su serie; tachada si está oculta. Devuelve la zona clicable."""
        on = self.visible[key]
        color = COLORS[key] if on else THEME.text_faint
        if key == "actual":
            color = THEME.text_muted if on else THEME.text_faint
            x, y = pos
            pygame.draw.rect(surface, COLORS["actual"] if on else (50, 47, 44), (x, y - px(6), px(4), px(12)))
            pos = (x + px(10), y)
        rect = ui.blit_text(surface, LABELS[key], pos, THEME.size_label, color, weight="semibold", anchor=anchor)
        if suffix:
            value = ui.blit_text(surface, suffix, (rect.right, rect.centery), THEME.size_label,
                                 THEME.text if on else THEME.text_faint, anchor="midleft")
            rect = rect.union(value)
        if not on:
            pygame.draw.line(surface, THEME.text_faint, (rect.left, rect.centery), (rect.right, rect.centery))
        hit = rect.inflate(px(10), px(8))
        if key == "actual":
            hit.left -= px(10)
        if hit.collidepoint(self.mouse):
            ui.hairline(surface, rect.left, rect.right, rect.bottom + px(1), (*THEME.text, 90))
        return hit

    def draw_tooltip(self, surface, hover) -> None:
        """Tooltip discreto: fondo sólido simple, sin sombra; línea guía vertical."""
        v, s = self.view, self.series_idx
        if hover == "next":
            x = self.next_x
            lstm_n, ma_n, hist_n = v["next"][s]
            title = f"Próxima semana · {fmt_range(v['next_start'], v['next_end'])}"
            rows = [("Real", "aún no observado", THEME.text_muted),
                    ("LSTM", f"{fmt(lstm_n)}  (± {fmt(v['val_mae'][s])})", COLORS["lstm"]),
                    ("Media móvil", fmt(ma_n), COLORS["moving_average"]),
                    ("Media histórica", fmt(hist_n), COLORS["hist_mean"])]
        else:
            w = hover
            x = self.x_center(w)
            start = self.results["weekly"].index[w]
            title = f"Semana {w + 1} · {fmt_range(start, start + pd.Timedelta(6, unit='D'))}"
            rows = [("Real", fmt(v["actual"][w, s], 0), THEME.text),
                    ("LSTM", fmt(v["lstm"][w, s]), COLORS["lstm"]),
                    ("Media móvil", fmt(v["moving_average"][w, s]), COLORS["moving_average"]),
                    ("Media histórica", fmt(v["hist_mean"][w, s]), COLORS["hist_mean"]),
                    ("Tramo", PART_NAMES[self.part_of(w)], THEME.text_muted)]
            ui.vhairline(surface, int(x), self.plot.top, self.plot.bottom, (*THEME.text, 80))
        title_img = ui.render_text(title, THEME.size_label, THEME.text, weight="semibold")
        row_h = px(21)
        box = pygame.Rect(0, 0, max(title_img.get_width(), px(200)) + px(28), px(38) + row_h * len(rows))
        box.topleft = (x + px(14), self.mouse[1] - box.height // 2)
        if box.right > self.score.left - px(12):
            box.right = int(x) - px(14)
        box.clamp_ip(pygame.Rect(0, self.plot.top - px(20), self.size[0], self.plot.height + px(40)))
        solid_box(surface, box)
        surface.blit(title_img, (box.left + px(14), box.top + px(10)))
        y = box.top + px(36)
        for label, value, color in rows:
            ui.aa_line(surface, color, (box.left + px(14), y + px(9)), (box.left + px(24), y + px(9)), px(2))
            ui.blit_text(surface, label, (box.left + px(32), y), THEME.size_label, THEME.text_muted)
            ui.blit_text(surface, value, (box.right - px(14), y), THEME.size_label, color, weight="semibold",
                         anchor="topright")
            y += row_h

    def part_of(self, week: int) -> str:
        ckpt = self.results["ckpt"]
        if week < ckpt["train_end"]:
            return "train"
        return "val" if week < ckpt["val_end"] else "test"

    # --- marcador tipográfico (sin caja) ---
    def draw_scoreboard(self, surface) -> None:
        sb, v = self.score, self.view
        metrics = self.results["metrics"].set_index("series")
        x0, right = sb.left, sb.right
        x_mae = right - px(58)
        ui.blit_eyebrow(surface, "Error en prueba", (x0, sb.top))
        # Selector como dos palabras; subrayado de acento en la activa.
        x = x0
        self.segment_rects = {}
        for scope, label in (("series", "Esta serie"), ("global", "Global")):
            active = scope == self.scope
            r = ui.blit_text(surface, label, (x, sb.top + px(28)), THEME.size_body,
                             THEME.text if active else THEME.text_faint, weight="semibold" if active else "medium")
            hit = r.inflate(px(12), px(10))
            if active:
                pygame.draw.rect(surface, THEME.accent, (r.left, r.bottom + px(3), r.width, max(2, px(2))))
            elif hit.collidepoint(self.mouse):
                ui.hairline(surface, r.left, r.right, r.bottom + px(3), (*THEME.text, 110))
            self.segment_rects[scope] = hit
            x = r.right + px(22)
        ui.blit_text(surface, "8 semanas · menor es mejor", (x0, sb.top + px(60)), THEME.size_caption,
                     THEME.text_faint)
        y = sb.top + px(90)
        ui.blit_text(surface, "MAE", (x_mae, y), THEME.size_caption, THEME.text_faint, weight="semibold",
                     anchor="topright")
        ui.blit_text(surface, "RMSE", (right, y), THEME.size_caption, THEME.text_faint, weight="semibold",
                     anchor="topright")
        y += px(24)
        block = metrics.loc[self.series_name if self.scope == "series" else "GLOBAL"]
        best = {col: min(block[f"{key}_{col}"] for _, key, _ in MODEL_ROWS) for col in ("mae", "rmse")}
        row_h = px(38)
        ui.hairline(surface, x0, right, y)
        for label, key, color_key in MODEL_ROWS:
            reference = key == "naive"  # la ingenua es la referencia mínima: más tenue
            mid = y + row_h // 2
            mark = COLORS[color_key]
            if key == "moving_average":
                ui.dashed_polyline(surface, mark, [(x0, mid), (x0 + px(14), mid)], px(2), px(4), px(3))
            else:
                ui.aa_line(surface, mark, (x0, mid), (x0 + px(14), mid), px(3 if key == "lstm" else 2))
            ui.blit_text(surface, label, (x0 + px(22), mid), THEME.size_label,
                         THEME.text_faint if reference else THEME.text, weight="medium", anchor="midleft")
            for col, xr in (("mae", x_mae), ("rmse", right)):
                value = block[f"{key}_{col}"]
                is_best = value == best[col]
                color = THEME.text if is_best else (THEME.text_faint if reference else THEME.text_muted)
                r = ui.blit_text(surface, fmt(value, 2), (xr, mid), THEME.size_body, color,
                                 weight="semibold" if is_best else "regular", anchor="midright")
                if is_best:  # el mejor de cada columna: subrayado de acento
                    pygame.draw.rect(surface, THEME.accent, (r.left, r.bottom + px(1), r.width, max(2, px(2))))
            y += row_h
            ui.hairline(surface, x0, right, y)
        # Nota de significancia: una frase en serif cursiva.
        y += px(14)
        for line in ui.wrap_text(v["significance_note"], THEME.size_body + 1, sb.width, weight="serif_italic"):
            ui.blit_text(surface, line, (x0, y), THEME.size_body + 1, THEME.text_muted, weight="serif_italic")
            y += px(21)

    # --- tira de mini-gráficas (selector de serie) ---
    def sparkline(self, s: int, active: bool, size) -> pygame.Surface:
        """Mini-gráfica de las 52 semanas + punto de la próxima semana (pre-renderizada y cacheada)."""
        key = (s, active, size)
        if key not in self.sparks:
            w, h = size
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            values = self.view["actual"][:, s]
            top = max(float(np.nanmax(self.view["actual"])), 1.0)  # misma escala para las 6: se comparan niveles
            n = len(values)
            dot_x = w - px(5)
            xs = np.linspace(px(1), dot_x - px(12), n)
            ys = h - px(3) - values / top * (h - px(8))
            color = THEME.text if active else (95, 89, 84)
            ui.aa_polyline(surf, color, list(zip(xs, ys)), px(1.5) if active else 1)
            nxt = self.view["next"][s][0]
            ui.aa_circle(surf, THEME.accent if active else THEME.text_faint,
                         (dot_x, h - px(3) - nxt / top * (h - px(8))), px(3.5))
            self.sparks[key] = ui._to_display_format(surf)
        return self.sparks[key]

    def draw_strip(self, surface) -> None:
        v = self.view
        self.name_rects = []
        for i, rect in enumerate(self.strip):
            active = i == self.series_idx
            hovered = rect.collidepoint(self.mouse) and not active
            name_color = THEME.accent_text if active else (THEME.text_muted if hovered else THEME.text_faint)
            ui.blit_text(surface, v["names"][i], rect.topleft, THEME.size_label, name_color,
                         weight="semibold" if active else "medium")
            # Pronóstico de la próxima semana a la derecha de su mini-gráfica (no compite con el nombre).
            spark_rect = pygame.Rect(rect.left, rect.top + px(22), rect.width - px(44), rect.height - px(22))
            ui.blit_text(surface, fmt(v["next"][i][0]), (rect.right, spark_rect.centery), THEME.size_label,
                         THEME.accent_text if active else THEME.text_faint,
                         weight="semibold" if active else "regular", anchor="midright")
            surface.blit(self.sparkline(i, active or hovered, spark_rect.size), spark_rect)
            self.name_rects.append(rect.inflate(px(10), px(10)))

    # --- ayudas emergentes y cursor ---
    def info_paragraphs(self) -> list[str]:
        v, s = self.view, self.series_idx
        total = v["test_per_series"] * len(v["names"])
        return [
            f"La LSTM empieza en la semana {v['first_lstm_week'] + 1}: necesita {v['first_lstm_week']} semanas "
            f"previas. La media móvil empieza en la {v['first_ma_week'] + 1}.",
            f"Banda = LSTM ± MAE de validación ({fmt(v['val_mae'][s])} u). Semanas de prueba dentro de la banda: "
            f"{v['inside'][s]} de {v['test_per_series']} (las 6 series: {v['inside_total']} de {total}).",
            "Clic en una etiqueta (LSTM, Media móvil, Media histórica, Ventas reales) para mostrarla u ocultarla.",
        ]

    def draw_info_tooltip(self, surface) -> None:
        width = px(420)
        paragraphs = [ui.wrap_text(t, THEME.size_caption, width - px(32)) for t in self.info_paragraphs()]
        height = px(22) + sum(px(19) * len(lines) for lines in paragraphs) + px(10) * (len(paragraphs) - 1)
        box = pygame.Rect(0, 0, width, height)
        box.topleft = (self.info_rect.left, self.info_rect.bottom + px(10))
        box.clamp_ip(surface.get_rect().inflate(-px(16), -px(16)))
        solid_box(surface, box)
        y = box.top + px(11)
        for lines in paragraphs:
            for line in lines:
                ui.blit_text(surface, line, (box.left + px(16), y), THEME.size_caption, THEME.text_muted)
                y += px(19)
            y += px(10)

    def update_cursor(self) -> None:
        """Cursor de mano sobre lo que se puede pulsar."""
        clickable = [*self.legend_rects.values(), *self.name_rects, *self.segment_rects.values(),
                     self.export_button.rect, self.back_button.rect]
        hand = any(r.collidepoint(self.mouse) for r in clickable) or self.slider.dragging
        if hand != self._cursor:
            self._cursor = hand
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hand else pygame.SYSTEM_CURSOR_ARROW)
            except pygame.error:  # sin ventana real (p. ej. pruebas headless)
                pass


def run(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    try:
        view = ForecastScreen(load_cached(), size=screen.get_size())
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        view = ForecastScreen(None, str(exc), size=screen.get_size())
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
