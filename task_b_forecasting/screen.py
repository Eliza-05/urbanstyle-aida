"""Tarea B (Elizabeth): pantalla Pygame del pronóstico de demanda.

Muestra real vs. predicho (LSTM y media móvil) de la serie elegida con el slider, sus métricas en
prueba, el pronóstico de la próxima semana y un botón para exportar las predicciones a CSV.
Solo CARGA el checkpoint ya entrenado (models/lstm_forecast.pt); nunca entrena.
"""
import pygame

from common import config
from common.ui import Button, Slider

# Colores propios del gráfico (el resto viene de common/config.py).
COLOR_REAL = (230, 230, 235)
COLOR_LSTM = (90, 160, 255)
COLOR_MA = (255, 160, 60)
COLOR_GRID = (55, 60, 72)
COLOR_TEST = (44, 48, 60)
COLOR_MUTED = (160, 165, 180)

PLOT = pygame.Rect(70, 100, 620, 330)   # área del gráfico
Y_MAX = 25                              # unidades/semana (máximo real = 24)
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

_cache = {}  # resultados cargados una sola vez por sesión


def load_cached():
    """Carga (una vez) el checkpoint y las predicciones. Import diferido: torch solo se carga al entrar."""
    if "results" not in _cache:
        from task_b_forecasting.evaluate import load_results
        _cache["results"] = load_results()
    return _cache["results"]


class ForecastScreen:
    def __init__(self, results=None, error: str | None = None):
        self.results = results
        self.error = error
        self.font_title = pygame.font.SysFont(None, 34)
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 19)
        self.series_idx = 0
        self.slider = Slider((PLOT.left, 520, PLOT.width, 8), 0, 5, 0)
        self.export_button = Button((730, 520, 270, 44), "Exportar CSV", self.font)
        self.back_button = Button((730, 575, 270, 44), "Volver al menú (ESC)", self.font)
        self.status = ""
        self.done = False

    # ---------- eventos ----------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.done = True
        elif self.back_button.handle_event(event):
            self.done = True
        elif self.results is None:
            return
        elif self.export_button.handle_event(event):
            from task_b_forecasting.evaluate import export_predictions
            path = export_predictions(self.results)
            self.status = f"Exportado: {path.relative_to(config.ROOT).as_posix()}"
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
            step = 1 if event.key == pygame.K_RIGHT else -1
            self.set_series(self.series_idx + step)
        else:
            value = self.slider.handle_event(event)
            self.set_series(round(value))

    def set_series(self, idx: int) -> None:
        self.series_idx = max(0, min(5, idx))
        if not self.slider.dragging:
            self.slider.value = self.series_idx  # el slider "salta" a la serie elegida

    @property
    def series_name(self) -> str:
        return self.results["ckpt"]["series_names"][self.series_idx]

    # ---------- dibujo ----------
    def text(self, surface, msg, pos, font=None, color=config.COLOR_TEXT, anchor="topleft"):
        img = (font or self.font).render(msg, True, color)
        rect = img.get_rect(**{anchor: pos})
        surface.blit(img, rect)
        return rect

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(config.COLOR_BG)
        self.text(surface, "Tarea B · Pronóstico de demanda semanal (LSTM)", (20, 14), self.font_title)
        self.text(surface, "Dataset: 3 categorías originales × género = 6 series · unidades vendidas por semana, 2023",
                  (20, 50), self.font_small, COLOR_MUTED)
        if self.results is None:
            self.text(surface, "No se pudo cargar el modelo entrenado:", (20, 120))
            self.text(surface, self.error or "", (20, 150), self.font_small, COLOR_MUTED)
            self.text(surface, "Ejecuta: python -m task_b_forecasting.train", (20, 190))
            self.back_button.draw(surface)
            return
        self.draw_plot(surface)
        self.draw_slider(surface)
        self.draw_panel(surface)
        self.export_button.draw(surface)
        self.back_button.draw(surface)

    def to_xy(self, week: float, units: float) -> tuple[int, int]:
        n = self.results["ckpt"]["n_weeks"]  # semanas 0..n-1 históricas + semana n = próxima
        x = PLOT.left + (week + 0.5) / (n + 1) * PLOT.width  # deja medio espacio a cada lado
        y = PLOT.bottom - min(units, Y_MAX) / Y_MAX * PLOT.height
        return int(x), int(y)

    def draw_plot(self, surface) -> None:
        ckpt, weekly = self.results["ckpt"], self.results["weekly"]
        name = self.series_name
        self.text(surface, f"Serie: {name}  (unidades vendidas por semana)", (PLOT.left, 74), self.font)

        # Zona de prueba sombreada + rejilla horizontal con etiquetas.
        test_left = self.to_xy(ckpt["val_end"] - 0.5, 0)[0]
        pygame.draw.rect(surface, COLOR_TEST, (test_left, PLOT.top, PLOT.right - test_left, PLOT.height))
        self.text(surface, "prueba", (test_left + 4, PLOT.top + 2), self.font_small, COLOR_MUTED)
        for u in range(0, Y_MAX + 1, 5):
            y = self.to_xy(0, u)[1]
            pygame.draw.line(surface, COLOR_GRID, (PLOT.left, y), (PLOT.right, y))
            self.text(surface, str(u), (PLOT.left - 8, y), self.font_small, COLOR_MUTED, "midright")
        # Etiquetas de mes en el eje x.
        last_month = None
        for w, date in enumerate(weekly.index):
            if date.month != last_month:
                last_month = date.month
                x = self.to_xy(w, 0)[0]
                pygame.draw.line(surface, COLOR_GRID, (x, PLOT.bottom), (x, PLOT.bottom + 5))
                self.text(surface, MONTHS[date.month - 1], (x, PLOT.bottom + 7), self.font_small, COLOR_MUTED, "midtop")
        pygame.draw.rect(surface, COLOR_GRID, PLOT, 1)

        # Real (todas las semanas), LSTM y media móvil (semanas con historia suficiente).
        real = [self.to_xy(w, v) for w, v in enumerate(weekly[name])]
        pygame.draw.lines(surface, COLOR_REAL, False, real, 1)
        for p in real:
            pygame.draw.circle(surface, COLOR_REAL, p, 2)

        pred = self.results["predictions"]
        pred = pred[pred["series"] == name]
        week_pos = {d.date(): i for i, d in enumerate(weekly.index)}
        weeks = [week_pos[d] for d in pred["week_start"]]
        ma_pts = [self.to_xy(w, v) for w, v in zip(weeks, pred["moving_average"])]
        self.dashed(surface, COLOR_MA, ma_pts)
        lstm_pts = [self.to_xy(w, v) for w, v in zip(weeks, pred["lstm"])]
        pygame.draw.lines(surface, COLOR_LSTM, False, lstm_pts, 3)

        # Próxima semana: punto de la LSTM y de la media móvil.
        nxt = self.next_week_row()
        n = ckpt["n_weeks"]
        pygame.draw.circle(surface, COLOR_MA, self.to_xy(n, nxt["moving_average"]), 6, 2)
        pygame.draw.circle(surface, COLOR_LSTM, self.to_xy(n, nxt["lstm"]), 7)

        # Leyenda.
        x, y = PLOT.left, PLOT.bottom + 30
        for label, color, style in (("real", COLOR_REAL, "line"), ("LSTM", COLOR_LSTM, "thick"),
                                    (f"media móvil (k={ckpt['ma_k']})", COLOR_MA, "dash"),
                                    ("próxima semana", COLOR_LSTM, "dot")):
            if style == "dot":
                pygame.draw.circle(surface, color, (x + 12, y + 8), 6)
            elif style == "dash":
                self.dashed(surface, color, [(x, y + 8), (x + 24, y + 8)])
            else:
                pygame.draw.line(surface, color, (x, y + 8), (x + 24, y + 8), 3 if style == "thick" else 1)
            x = self.text(surface, label, (x + 30, y), self.font_small).right + 22

    @staticmethod
    def dashed(surface, color, points, dash: int = 6) -> None:
        """Línea discontinua (pygame no la trae): dibuja tramos alternos de cada segmento."""
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            length = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
            for start in range(0, length, dash * 2):
                end = min(start + dash, length)
                a = (x1 + (x2 - x1) * start / length, y1 + (y2 - y1) * start / length)
                b = (x1 + (x2 - x1) * end / length, y1 + (y2 - y1) * end / length)
                pygame.draw.line(surface, color, a, b, 2)

    def draw_slider(self, surface) -> None:
        self.text(surface, "Elige la serie (arrastra el slider o usa las flechas izquierda/derecha):",
                  (PLOT.left, 490), self.font_small, COLOR_MUTED)
        self.slider.draw(surface)
        names = self.results["ckpt"]["series_names"]
        for i, name in enumerate(names):
            x = PLOT.left + i / 5 * PLOT.width
            cat, gender = name.split("-")
            color = config.COLOR_TEXT if i == self.series_idx else COLOR_MUTED
            anchor = "midtop" if 0 < i < 5 else ("topleft" if i == 0 else "topright")
            self.text(surface, cat, (x, 540), self.font_small, color, anchor)
            self.text(surface, gender, (x, 556), self.font_small, color, anchor)

    def next_week_row(self):
        nxt = self.results["next_week"]
        return nxt[nxt["series"] == self.series_name].iloc[0]

    def draw_panel(self, surface) -> None:
        metrics = self.results["metrics"].set_index("series")
        k = self.results["ckpt"]["ma_k"]
        x, y = 730, 90
        self.text(surface, "Error en prueba (últimas 8 semanas)", (x, y), self.font)
        y += 30
        for col, label in ((x + 150, "MAE"), (x + 215, "RMSE")):
            self.text(surface, label, (col, y), self.font_small, COLOR_MUTED)
        rows = (("LSTM", "lstm", COLOR_LSTM), (f"Media móvil k={k}", "moving_average", COLOR_MA),
                ("Ingenua", "naive", COLOR_MUTED))
        for title, block in ((self.series_name, metrics.loc[self.series_name]), ("Global (6 series)", metrics.loc["GLOBAL"])):
            y += 22
            self.text(surface, title, (x, y), self.font_small)
            for label, key, color in rows:
                y += 20
                self.text(surface, label, (x + 8, y), self.font_small, color)
                self.text(surface, f"{block[f'{key}_mae']:.2f}", (x + 150, y), self.font_small)
                self.text(surface, f"{block[f'{key}_rmse']:.2f}", (x + 215, y), self.font_small)
            y += 8
        self.text(surface, "unidades por semana; menor es mejor", (x, y + 14), self.font_small, COLOR_MUTED)

        nxt = self.next_week_row()
        mae_lstm = metrics.loc[self.series_name, "lstm_mae"]
        y += 50
        self.text(surface, f"Próxima semana ({nxt['week_start']:%d-%m-%Y})", (x, y), self.font)
        y += 28
        self.text(surface, f"LSTM: {nxt['lstm']:.1f} u  (± {mae_lstm:.1f} MAE)", (x + 8, y), self.font_small, COLOR_LSTM)
        y += 20
        self.text(surface, f"Media móvil: {nxt['moving_average']:.1f} u", (x + 8, y), self.font_small, COLOR_MA)
        if self.status:
            self.text(surface, self.status, (730, 498), self.font_small, COLOR_MUTED)


def run(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    try:
        view = ForecastScreen(load_cached())
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        view = ForecastScreen(None, str(exc))
    while not view.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.event.post(pygame.event.Event(pygame.QUIT))  # el menú también debe cerrarse
                return
            view.handle_event(event)
        view.draw(screen)
        pygame.display.flip()
        clock.tick(config.FPS)
