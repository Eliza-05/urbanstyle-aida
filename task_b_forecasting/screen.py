"""Tarea B (Elizabeth): pantalla Pygame del pronóstico de demanda (estilo dashboard).

En 5 segundos debe leerse: cuánto se vendió cada semana (barras), qué predice cada modelo (líneas),
qué tan bueno es cada uno (KPI y tabla de error en prueba) y cuánto se venderá la próxima semana.
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
LEGEND = (                          # (clave, etiqueta) en el orden de la leyenda
    ("actual", "Ventas reales"),
    ("lstm", "LSTM ± MAE val."),
    ("moving_average", "Media móvil (8 sem.)"),
    ("hist_mean", "Media histórica"),
)
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
        return "LSTM, media móvil y media histórica: sin diferencia significativa (bootstrap 95 %)"
    return "Ver significancia en outputs/task_b/metrics/bootstrap.csv."


def nice_ceiling(value: float) -> float:
    """Máximo del eje Y redondeado hacia arriba a un múltiplo de 5."""
    return max(10.0, math.ceil(value / 5.0) * 5.0)


def fmt(value: float, decimals: int = 1) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{decimals}f}".replace(".", ",")


def fmt_date(ts: pd.Timestamp, year: bool = False) -> str:
    text = f"{ts.day:02d}-{MONTHS[ts.month - 1]}"
    return f"{text}-{ts.year}" if year else text


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
        self.visible = {key: True for key, _ in LEGEND}
        self.mouse = (-1, -1)
        self.toast = ui.Toast()
        self.fade = ui.FadeIn()
        self.legend_rects, self.name_rects = {}, []   # zonas clicables; se actualizan al dibujar
        self.info_rect = pygame.Rect(0, 0, 0, 0)       # ícono ⓘ junto al título de la gráfica
        self.scope = "series"                          # tabla de métricas: "series" o "global"
        self._cursor = None
        self.layout()
        self.back_button = ui.Button(self.back_rect, "← Menú (Esc)")
        if results is None:
            return
        self.view = results["view"]
        self.shown = self.target_vector(0)
        self.anim_from = self.anim_to = self.shown
        self.anim_start = -ANIM_MS
        self.slider = ui.Slider(self.slider_rect, 0, len(self.view["names"]) - 1, 0)
        self.knob_target = 0.0
        self.export_button = ui.Button(self.export_rect, "Exportar CSV", primary=True)
        # Capas con alfa reutilizadas en cada cuadro (crearlas cuesta; limpiarlas es barato).
        self.plot_layer = ui._to_display_format(pygame.Surface(self.plot_area.size, pygame.SRCALPHA))
        self.band_layer = ui._to_display_format(pygame.Surface(self.plot_area.size, pygame.SRCALPHA))
        self.band_gradient = ui.gradient_surface(self.plot_area.size, COLORS["lstm"], COLORS["lstm"],
                                                 vertical=True, alpha=(55, 10))
        n = len(self.view["actual"])
        self.xs = self.plot_hist.left + (np.arange(n) + 0.5) * self.plot_hist.width / n  # centro de cada semana
        self.static = ui._to_display_format(self.render_static())
        # Búfer con lo que solo cambia con el estado (KPI, panel, gráfica): se redibuja cuando cambia su clave.
        self.buffer = self.static.copy()
        self._side_key = self._chart_key = None

    # ---------- geometría (en px lógicos -> físicos) ----------
    def layout(self) -> None:
        W, H = self.size[0] / THEME.scale, self.size[1] / THEME.scale   # tamaño lógico
        m, gap = 16, 12
        panel_w = 300 if W < 1200 else 340
        top = 78
        left_w = W - 2 * m - panel_w - gap
        self.back_rect = R(W - m - 140, 18, 140, 38)
        self.export_rect = R(W - m - 140 - 10 - 150, 18, 150, 38)
        self.panel = R(W - m - panel_w, top, panel_w, H - top - m)
        kpi_h = 78
        kpi_w = (left_w - 2 * gap) / 3
        self.kpis = [R(m + i * (kpi_w + gap), top, kpi_w, kpi_h) for i in range(3)]
        slider_h = 100
        self.slider_card = R(m, H - m - slider_h, left_w, slider_h)
        chart_top = top + kpi_h + gap
        self.chart = R(m, chart_top, left_w, H - m - slider_h - gap - chart_top)
        c = self.chart
        self.plot = pygame.Rect(c.left + px(46), c.top + px(72), c.width - px(46 + 18), c.height - px(72 + 36))
        next_w = px(86)
        self.plot_hist = pygame.Rect(self.plot.left, self.plot.top, self.plot.width - next_w - px(10), self.plot.height)
        self.plot_next = pygame.Rect(self.plot_hist.right + px(10), self.plot.top, next_w, self.plot.height)
        self.plot_area = self.plot.inflate(0, px(2))  # capa de barras, zonas y cuadrícula
        sc = self.slider_card
        self.slider_rect = pygame.Rect(sc.left + px(76), sc.top + px(40), sc.width - px(152), px(8))
        seg = pygame.Rect(self.panel.left + px(18), self.panel.top + px(72), self.panel.width - px(36), px(36))
        half = seg.width // 2
        self.segment_rects = {"series": pygame.Rect(seg.left, seg.top, half, seg.height),
                              "global": pygame.Rect(seg.left + half, seg.top, seg.width - half, seg.height)}
        self.segment_track = seg

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
        """Clic en la leyenda (mostrar/ocultar) o en el nombre de una serie. True si lo atendió."""
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
        """Lo que no cambia nunca se dibuja una sola vez: fondo, encabezado, tarjetas y textos fijos."""
        layer = ui.background(self.size).copy()
        title = ui.blit_text(layer, "Pronóstico de demanda semanal", (px(16), px(14)), THEME.size_title,
                             weight="bold")
        ui.pill(layer, "TAREA B · LSTM", (title.right + px(12), title.centery + px(1)), anchor="midleft")
        ui.blit_text(layer, "3 categorías originales × género = 6 series  ·  unidades vendidas por semana, 2023",
                     (px(16), title.bottom + px(6)), THEME.size_label, THEME.text_muted)
        for rect in (*self.kpis, self.chart, self.slider_card, self.panel):
            ui.draw_card(layer, rect)
        sc = self.slider_card
        head = ui.blit_text(layer, "Serie", (sc.left + px(18), sc.top + px(12)), THEME.size_heading,
                            weight="semibold")
        ui.blit_text(layer, "Arrastra o usa ← →",
                     (head.right + px(14), head.centery + px(1)), THEME.size_caption, THEME.text_faint,
                     anchor="midleft")
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
        self.draw_slider(surface)
        if self.info_rect.collidepoint(self.mouse):
            self.draw_info_tooltip(surface)
        elif any(r.collidepoint(self.mouse) for r in self.legend_rects.values()):
            self.draw_hint(surface, "Clic para mostrar u ocultar")
        self.update_cursor()
        self.export_button.draw(surface)
        self.back_button.draw(surface)
        self.toast.draw(surface)
        self.fade.draw(surface)

    def refresh_buffer(self) -> None:
        """Redibuja en el búfer solo las zonas cuyo estado cambió (60 FPS sin recalcular todo)."""
        seg_hover = next((k for k, r in self.segment_rects.items() if r.collidepoint(self.mouse)), None)
        side_key = (self.series_idx, self.scope, seg_hover)
        if self._side_key != side_key:
            for rect in (*self.kpis, self.panel):
                self.buffer.blit(self.static, rect, rect)
            self.draw_kpis(self.buffer)
            self.draw_panel(self.buffer)
            self._side_key = side_key
        animating = pygame.time.get_ticks() - self.anim_start < ANIM_MS + 20
        legend_hover = next((k for k, r in self.legend_rects.items() if r.collidepoint(self.mouse)), None)
        key = (self.series_idx, tuple(self.visible.values()), self.hover_week(), legend_hover,
               self.info_rect.collidepoint(self.mouse))
        if animating or key != self._chart_key:
            self.buffer.blit(self.static, self.chart, self.chart)
            self.draw_chart(self.buffer, self.unpack(self.shown))
            self._chart_key = None if animating else key

    def draw_error(self, surface) -> None:
        surface.blit(ui.background(self.size), (0, 0))
        ui.blit_text(surface, "Pronóstico de demanda semanal", (px(16), px(14)), THEME.size_title, weight="bold")
        ui.blit_text(surface, "No se pudo cargar el modelo entrenado:", (px(16), px(90)), THEME.size_heading,
                     weight="semibold")
        ui.blit_text(surface, self.error or "", (px(16), px(118)), THEME.size_label, THEME.text_muted)
        ui.blit_text(surface, "Ejecuta:  python -m task_b_forecasting.train", (px(16), px(150)), THEME.size_body)
        self.back_button.draw(surface)
        self.fade.draw(surface)

    # --- tarjetas KPI ---
    def draw_kpis(self, surface) -> None:
        v, s = self.view, self.series_idx
        block = self.results["metrics"].set_index("series").loc[self.series_name]
        last_start = self.results["weekly"].index[-1]
        last_end = last_start + pd.Timedelta(6, unit="D")
        cards = (
            ("Próxima semana · LSTM", COLORS["lstm"], f"{fmt(v['next'][s][0])} u", f"± {fmt(v['val_mae'][s])}",
             THEME.text),
            ("Ventas última semana", COLORS["actual"], f"{fmt(v['actual'][-1, s], 0)} u",
             f"{last_start.day}–{last_end.day} {MONTHS[last_end.month - 1]}", THEME.text),
            ("MAE LSTM en prueba", COLORS["lstm"], fmt(block["lstm_mae"], 2), "u/semana", THEME.text),
        )
        for rect, (label, dot, value, extra, value_color) in zip(self.kpis, cards):
            x = rect.left + px(18)
            ui.aa_circle(surface, dot, (x + px(4), rect.top + px(21)), px(4))
            ui.blit_text(surface, label, (x + px(14), rect.top + px(21)), THEME.size_label, THEME.text_muted,
                         weight="medium", anchor="midleft")
            # El número grande se achica si no cabe junto a su texto secundario (nada toca el borde).
            available = rect.width - px(36) - px(10) - ui.render_text(extra, THEME.size_label).get_width()
            size = THEME.size_kpi
            while size > THEME.size_heading and ui.render_text(value, size, weight="bold").get_width() > available:
                size -= 1
            num = ui.blit_text(surface, value, (x, rect.bottom - px(12)), size, value_color, weight="bold",
                               anchor="bottomleft")
            ui.blit_text(surface, extra, (num.right + px(10), num.bottom - px(4)), THEME.size_label,
                         THEME.text_muted, anchor="bottomleft")

    # --- gráfica ---
    def x_center(self, week: int) -> float:
        return float(self.xs[week])

    def y_of(self, units: float, ymax: float) -> float:
        return self.plot.bottom - units / ymax * self.plot.height

    def hover_week(self):
        """Semana bajo el mouse: 0..51, "next" en la zona de próxima semana, o None."""
        x, y = self.mouse
        if not self.plot.top <= y <= self.plot.bottom:
            return None
        if self.plot_next.left <= x <= self.plot_next.right:
            return "next"
        if self.plot_hist.left <= x < self.plot_hist.right:
            n = len(self.view["actual"])
            return int((x - self.plot_hist.left) / self.plot_hist.width * n)
        return None

    def draw_chart(self, surface, d) -> None:
        v, s = self.view, self.series_idx
        n = len(v["actual"])
        ymax = d["ymax"]
        c = self.chart
        hover = self.hover_week()

        # Encabezado de la tarjeta y leyenda interactiva.
        head = ui.blit_text(surface, self.series_name, (c.left + px(18), c.top + px(14)), THEME.size_heading,
                            weight="semibold")
        self.info_rect = pygame.Rect(0, 0, px(22), px(22))
        self.info_rect.midleft = (head.right + px(10), head.centery)
        active = self.info_rect.collidepoint(self.mouse)
        ui.aa_circle(surface, THEME.accent if active else (60, 52, 110), self.info_rect.center, px(10))
        ui.blit_text(surface, "i", self.info_rect.center, THEME.size_caption, (255, 255, 255), weight="bold",
                     anchor="center")
        self.draw_legend(surface, (c.left + px(18), c.top + px(42)))

        # Capa translúcida: zonas, cuadrícula y barras (pygame.draw ignora el alfa en la ventana).
        layer, origin = self.plot_layer, self.plot_area.topleft
        layer.fill((0, 0, 0, 0))
        local = lambda rect: rect.move(-origin[0], -origin[1])
        val_end = self.results["ckpt"]["val_end"]
        x_test = self.plot_hist.left + val_end * self.plot_hist.width / n
        zone = pygame.Rect(int(x_test), self.plot.top, self.plot_hist.right - int(x_test), self.plot.height)
        layer.blit(ui.rounded_surface(zone.size, px(8), (*THEME.accent, 22), (*THEME.accent, 60)), local(zone))
        next_zone = self.plot_next
        fill = (*THEME.accent, 40 if hover == "next" else 26)
        layer.blit(ui.rounded_surface(next_zone.size, px(8), fill, (*THEME.accent, 80)), local(next_zone))
        step = 5 if ymax <= 30 else 10
        for u in range(0, int(ymax) + 1, step):
            y = int(self.y_of(u, ymax)) - origin[1]
            pygame.draw.line(layer, THEME.grid, (0, y), (self.plot_hist.width + px(10) + next_zone.width, y))
        slot = self.plot_hist.width / n
        if self.visible["actual"]:
            radius = px(3)
            for w in range(n):
                top = self.y_of(d["actual"][w], ymax)
                bar = pygame.Rect(self.plot_hist.left + w * slot + px(1.5), top, max(2, slot - px(3)),
                                  self.plot.bottom - top)
                alpha = 240 if hover == w else 185
                pygame.draw.rect(layer, (*COLORS["actual"], alpha), local(bar),
                                 border_top_left_radius=radius, border_top_right_radius=radius)
        surface.blit(layer, origin)

        # Eje Y y meses.
        for u in range(0, int(ymax) + 1, step):
            ui.blit_text(surface, str(u), (self.plot.left - px(10), self.y_of(u, ymax)), THEME.size_caption,
                         THEME.text_faint, anchor="midright")
        weeks_idx = self.results["weekly"].index
        for w, date in enumerate(weeks_idx):
            if w == 0 or date.month != weeks_idx[w - 1].month:
                ui.blit_text(surface, MONTHS[date.month - 1], (self.x_center(w), self.plot.bottom + px(6)),
                             THEME.size_caption, THEME.text_faint, anchor="midtop")
        ui.pill(surface, "Prueba", (zone.centerx, self.plot.top + px(6)), anchor="midtop")

        # Banda ± MAE (relleno degradado) y líneas de los modelos, con antialias.
        weeks = np.arange(n)
        ok = np.isfinite(d["lstm"])
        lstm_pts = [(self.x_center(w), self.y_of(d["lstm"][w], ymax)) for w in weeks[ok]]
        if self.visible["lstm"]:
            band = d["band"]
            upper = [(self.x_center(w), self.y_of(min(ymax, d["lstm"][w] + band), ymax)) for w in weeks[ok]]
            lower = [(self.x_center(w), self.y_of(max(0, d["lstm"][w] - band), ymax)) for w in weeks[ok][::-1]]
            mask = self.band_layer
            mask.fill((0, 0, 0, 0))
            pygame.draw.polygon(mask, (255, 255, 255, 255), [(x - origin[0], y - origin[1]) for x, y in upper + lower])
            mask.blit(self.band_gradient, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            surface.blit(mask, origin)
            edge = tuple(int(a * 0.30 + b * 0.70) for a, b in zip(COLORS["lstm"], THEME.bg))
            ui.aa_polyline(surface, edge, upper, 1)
            ui.aa_polyline(surface, edge, lower[::-1], 1)
        if self.visible["moving_average"]:
            okm = np.isfinite(d["moving_average"])
            ui.dashed_polyline(surface, COLORS["moving_average"],
                               [(self.x_center(w), self.y_of(d["moving_average"][w], ymax)) for w in weeks[okm]],
                               px(2), px(7), px(5))
        if self.visible["hist_mean"]:
            okh = np.isfinite(d["hist_mean"])
            if okh.any():
                y = self.y_of(d["hist_mean"][okh][0], ymax)
                ui.aa_line(surface, COLORS["hist_mean"], (self.x_center(weeks[okh][0]) - slot / 2, y),
                           (self.plot_hist.right, y), px(2))
        if self.visible["lstm"]:
            ui.aa_polyline(surface, COLORS["lstm"], lstm_pts, px(3))

        self.draw_next_zone(surface, d, ymax)


    def draw_legend(self, surface, origin) -> None:
        x, y = origin
        self.legend_rects = {}
        for key, label in LEGEND:
            on = self.visible[key]
            color = COLORS[key] if on else THEME.text_faint
            sw = pygame.Rect(x, y + px(4), px(20), px(12))
            if key == "actual":
                surface.blit(ui.rounded_surface((px(10), sw.height), px(2), (*color, 200 if on else 120)),
                             (sw.left + px(5), sw.top))
            elif key == "lstm":
                if on:
                    surface.blit(ui.rounded_surface(sw.size, px(3), (*color, 60)), sw)
                ui.aa_line(surface, color, (sw.left, sw.centery), (sw.right, sw.centery), px(3))
            elif key == "moving_average":
                ui.dashed_polyline(surface, color, [(sw.left, sw.centery), (sw.right, sw.centery)], px(2), px(5), px(3))
            else:
                ui.aa_line(surface, color, (sw.left, sw.centery), (sw.right, sw.centery), px(2))
            text = ui.blit_text(surface, label, (sw.right + px(7), y), THEME.size_label,
                                THEME.text if on else THEME.text_faint, weight="medium")
            if not on:  # tachado: indica que la serie está oculta
                pygame.draw.line(surface, THEME.text_faint, (text.left, text.centery), (text.right, text.centery),
                                 max(1, px(1)))
            rect = pygame.Rect(x - px(6), y - px(4), text.right - x + px(12), text.height + px(8))
            if rect.collidepoint(self.mouse):
                surface.blit(ui.rounded_surface(rect.size, px(8), (255, 255, 255, 14), THEME.border), rect)
            self.legend_rects[key] = rect
            x = text.right + px(20)

    def draw_next_zone(self, surface, d, ymax) -> None:
        v, z = self.view, self.plot_next
        chip = ui.pill(surface, "Próxima", (z.centerx, z.top + px(6)), anchor="midtop")
        ui.blit_text(surface, fmt_date(v["next_start"]) + " –", (z.centerx, chip.bottom + px(6)),
                     THEME.size_caption, THEME.text_muted, anchor="midtop")
        ui.blit_text(surface, fmt_date(v["next_end"]), (z.centerx, chip.bottom + px(24)),
                     THEME.size_caption, THEME.text_muted, anchor="midtop")
        lstm_n, ma_n, hist_n = d["next"]
        if self.visible["moving_average"]:
            y = self.y_of(ma_n, ymax)
            ui.dashed_polyline(surface, COLORS["moving_average"], [(z.left + px(6), y), (z.right - px(6), y)],
                               px(2), px(5), px(4))
        if self.visible["hist_mean"]:
            y = self.y_of(hist_n, ymax)
            ui.aa_line(surface, COLORS["hist_mean"], (z.left + px(6), y), (z.right - px(6), y), px(2))
        if self.visible["lstm"]:
            cx = z.left + px(24)
            y, band = self.y_of(lstm_n, ymax), d["band"]
            y_hi, y_lo = self.y_of(min(ymax, lstm_n + band), ymax), self.y_of(max(0, lstm_n - band), ymax)
            ui.aa_line(surface, COLORS["lstm"], (cx, y_hi), (cx, y_lo), px(2))
            for yy in (y_hi, y_lo):
                ui.aa_line(surface, COLORS["lstm"], (cx - px(6), yy), (cx + px(6), yy), px(2))
            glow = ui.glow_surface(px(16), COLORS["lstm"], 120)
            surface.blit(glow, glow.get_rect(center=(cx, y)))
            ui.aa_circle(surface, COLORS["lstm"], (cx, y), px(6))
            ui.aa_circle(surface, THEME.bg, (cx, y), px(2.5))
            label = ui.render_text(fmt(lstm_n), THEME.size_label, COLORS["lstm"], weight="bold")
            box = label.get_rect(midleft=(cx + px(12), y)).inflate(px(10), px(4))
            surface.blit(ui.rounded_surface(box.size, box.height // 2, (*THEME.bg, 235), (*COLORS["lstm"], 160)), box)
            surface.blit(label, label.get_rect(center=box.center))

    def draw_tooltip(self, surface, hover) -> None:
        v, s = self.view, self.series_idx
        if hover == "next":
            x = self.plot_next.left + px(24)
            lstm_n, ma_n, hist_n = v["next"][s]
            title = f"Próxima semana · {fmt_date(v['next_start'])} – {fmt_date(v['next_end'], year=True)}"
            rows = [("Real", "aún no observado", THEME.text_muted),
                    ("LSTM", f"{fmt(lstm_n)}  (± {fmt(v['val_mae'][s])})", COLORS["lstm"]),
                    ("Media móvil", fmt(ma_n), COLORS["moving_average"]),
                    ("Media histórica", fmt(hist_n), COLORS["hist_mean"])]
        else:
            w = hover
            x = self.x_center(w)
            start = self.results["weekly"].index[w]
            title = f"Semana {w + 1} · {fmt_date(start)} – {fmt_date(start + pd.Timedelta(6, unit='D'), year=True)}"
            rows = [("Real", fmt(v["actual"][w, s], 0), THEME.text),
                    ("LSTM", fmt(v["lstm"][w, s]), COLORS["lstm"]),
                    ("Media móvil", fmt(v["moving_average"][w, s]), COLORS["moving_average"]),
                    ("Media histórica", fmt(v["hist_mean"][w, s]), COLORS["hist_mean"]),
                    ("Tramo", PART_NAMES[self.part_of(w)], THEME.text_muted)]
        ui.aa_line(surface, (*THEME.text, 120), (x, self.plot.top), (x, self.plot.bottom), 1)

        title_img = ui.render_text(title, THEME.size_label, THEME.text, weight="semibold")
        row_h = px(22)
        box = pygame.Rect(0, 0, max(title_img.get_width(), px(220)) + px(28), px(40) + row_h * len(rows))
        box.topleft = (x + px(16), self.mouse[1] - box.height // 2)
        if box.right > self.chart.right - px(8):
            box.right = int(x) - px(16)
        box.clamp_ip(self.chart.inflate(-px(10), -px(10)))
        ui.draw_card(surface, box, px(12), (16, 20, 33, 245), (255, 255, 255, 40))
        surface.blit(title_img, (box.left + px(14), box.top + px(12)))
        y = box.top + px(40)
        for label, value, color in rows:
            ui.aa_circle(surface, color, (box.left + px(19), y + px(9)), px(4))
            ui.blit_text(surface, label, (box.left + px(30), y), THEME.size_label, THEME.text_muted)
            ui.blit_text(surface, value, (box.right - px(14), y), THEME.size_label, color, weight="semibold",
                         anchor="topright")
            y += row_h

    def part_of(self, week: int) -> str:
        ckpt = self.results["ckpt"]
        if week < ckpt["train_end"]:
            return "train"
        return "val" if week < ckpt["val_end"] else "test"

    # --- slider ---
    def draw_slider(self, surface) -> None:
        self.slider.draw(surface)
        self.name_rects = []
        for i, name in enumerate(self.view["names"]):
            x = self.slider.x_of(i)
            cat, gender = name.split("-")
            selected = i == self.series_idx
            top = self.slider.rect.bottom + px(11)
            img1 = ui.render_text(cat, THEME.size_label, THEME.text if selected else THEME.text_muted,
                                  weight="semibold" if selected else "medium")
            img2 = ui.render_text(gender, THEME.size_caption, THEME.accent_text if selected else THEME.text_faint)
            rect = img1.get_rect(midtop=(x, top)).union(img2.get_rect(midtop=(x, top + img1.get_height())))
            rect = rect.inflate(px(18), px(6))
            if selected:
                surface.blit(ui.rounded_surface(rect.size, px(10), (*THEME.accent, 40), (*THEME.accent, 110)), rect)
            elif rect.collidepoint(self.mouse):
                surface.blit(ui.rounded_surface(rect.size, px(10), (255, 255, 255, 12), THEME.border), rect)
            surface.blit(img1, img1.get_rect(midtop=(x, top)))
            surface.blit(img2, img2.get_rect(midtop=(x, top + img1.get_height())))
            self.name_rects.append(rect)

    # --- panel derecho ---
    def draw_panel(self, surface) -> None:
        p, v = self.panel, self.view
        metrics = self.results["metrics"].set_index("series")
        pad = px(18)
        x0, inner_w = p.left + pad, p.width - 2 * pad
        x_rmse, x_mae = p.right - pad, p.right - pad - px(78)
        ui.blit_text(surface, "Error en prueba", (x0, p.top + px(16)), THEME.size_heading, weight="semibold")
        ui.blit_text(surface, "Últimas 8 semanas · menor es mejor", (x0, p.top + px(42)), THEME.size_caption,
                     THEME.text_faint)

        # Selector segmentado: una sola tabla, de esta serie o global.
        track = self.segment_track
        surface.blit(ui.rounded_surface(track.size, track.height // 2, (255, 255, 255, 10), THEME.border), track)
        for scope, label in (("series", "Esta serie"), ("global", "Global (6)")):
            rect = self.segment_rects[scope].inflate(-px(6), -px(6))
            if scope == self.scope:
                surface.blit(ui.gradient_surface(rect.size, THEME.accent, THEME.accent_2, rect.height // 2,
                                                 alpha=(150, 150)), rect)
                color, weight = (255, 255, 255), "semibold"
            else:
                if rect.collidepoint(self.mouse):
                    surface.blit(ui.rounded_surface(rect.size, rect.height // 2, (255, 255, 255, 16)), rect)
                color, weight = THEME.text_muted, "medium"
            ui.blit_text(surface, label, rect.center, THEME.size_label, color, weight=weight, anchor="center")

        block = metrics.loc[self.series_name if self.scope == "series" else "GLOBAL"]
        y = track.bottom + px(20)
        ui.blit_text(surface, self.series_name if self.scope == "series" else "6 series", (x0, y),
                     THEME.size_caption, THEME.text_faint)
        ui.blit_text(surface, "MAE", (x_mae, y), THEME.size_caption, THEME.text_muted, weight="semibold",
                     anchor="topright")
        ui.blit_text(surface, "RMSE", (x_rmse, y), THEME.size_caption, THEME.text_muted, weight="semibold",
                     anchor="topright")
        y += px(26)
        pygame.draw.line(surface, (40, 45, 62), (x0, y - px(4)), (p.right - pad, y - px(4)), max(1, px(1)))
        best = {col: min(block[f"{key}_{col}"] for _, key, _ in MODEL_ROWS) for col in ("mae", "rmse")}
        row_h = px(34)
        for label, key, color_key in MODEL_ROWS:
            reference = key == "naive"  # la ingenua es la referencia mínima: en tono más tenue
            label_color = THEME.text_faint if reference else THEME.text
            dot = COLORS[color_key]
            if reference:
                dot = tuple(int(c * 0.6 + b * 0.4) for c, b in zip(dot, THEME.bg))
            mid = y + row_h // 2
            ui.aa_circle(surface, dot, (x0 + px(5), mid), px(4))
            ui.blit_text(surface, label, (x0 + px(16), mid), THEME.size_label, label_color,
                         weight="regular" if reference else "medium", anchor="midleft")
            for col, xr in (("mae", x_mae), ("rmse", x_rmse)):
                value = block[f"{key}_{col}"]
                is_best = value == best[col]
                color = (255, 255, 255) if is_best else (THEME.text_faint if reference else THEME.text_muted)
                text = ui.render_text(fmt(value, 2), THEME.size_label, color,
                                      weight="semibold" if is_best else "regular")
                rect = text.get_rect(midright=(xr, mid))
                if is_best:  # mejor valor de la columna: resaltado en píldora
                    chip = rect.inflate(px(14), px(6))
                    chip.right = xr + px(5)
                    surface.blit(ui.rounded_surface(chip.size, chip.height // 2, (*THEME.accent, 70),
                                                    (*THEME.accent, 150)), chip)
                surface.blit(text, rect)
            y += row_h

        # Tarjeta de información: significancia (redactada desde bootstrap.csv).
        y += px(16)
        lines = ui.wrap_text(v["significance_note"], THEME.size_caption, inner_w - px(44))
        box = pygame.Rect(x0, y, inner_w, px(22) + px(19) * len(lines))
        surface.blit(ui.rounded_surface(box.size, px(12), (*THEME.accent, 22), (*THEME.accent, 70)), box)
        ui.aa_circle(surface, THEME.accent, (box.left + px(18), box.top + px(20)), px(8))
        ui.blit_text(surface, "i", (box.left + px(18), box.top + px(20)), THEME.size_caption, (255, 255, 255),
                     weight="bold", anchor="center")
        ty = box.top + px(11)
        for line in lines:
            ui.blit_text(surface, line, (box.left + px(36), ty), THEME.size_caption, THEME.text_muted)
            ty += px(19)

    # --- ayudas emergentes y cursor ---
    def info_paragraphs(self) -> list[str]:
        v, s = self.view, self.series_idx
        total = v["test_per_series"] * len(v["names"])
        return [
            f"La LSTM empieza en la semana {v['first_lstm_week'] + 1}: necesita {v['first_lstm_week']} semanas "
            f"previas. La media móvil empieza en la {v['first_ma_week'] + 1}.",
            f"Banda = LSTM ± MAE de validación ({fmt(v['val_mae'][s])} u). Semanas de prueba dentro de la banda: "
            f"{v['inside'][s]} de {v['test_per_series']} (las 6 series: {v['inside_total']} de {total}).",
        ]

    def draw_info_tooltip(self, surface) -> None:
        width = px(400)
        paragraphs = [ui.wrap_text(t, THEME.size_caption, width - px(32)) for t in self.info_paragraphs()]
        height = px(24) + sum(px(19) * len(lines) for lines in paragraphs) + px(10) * (len(paragraphs) - 1)
        box = pygame.Rect(0, 0, width, height)
        box.topleft = (self.info_rect.left - px(12), self.info_rect.bottom + px(10))
        box.clamp_ip(self.chart.inflate(-px(10), -px(10)))
        ui.draw_card(surface, box, px(12), (16, 20, 33, 248), (*THEME.accent, 120))
        y = box.top + px(12)
        for lines in paragraphs:
            for line in lines:
                ui.blit_text(surface, line, (box.left + px(16), y), THEME.size_caption, THEME.text_muted)
                y += px(19)
            y += px(10)

    def draw_hint(self, surface, text) -> None:
        """Pista corta junto al mouse (p. ej. sobre la leyenda)."""
        img = ui.render_text(text, THEME.size_caption, THEME.text, weight="medium")
        box = img.get_rect().inflate(px(20), px(10))
        box.topleft = (self.mouse[0] + px(14), self.mouse[1] + px(18))
        box.clamp_ip(surface.get_rect())
        ui.draw_card(surface, box, box.height // 2, (16, 20, 33, 245), THEME.border)
        surface.blit(img, img.get_rect(center=box.center))

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
