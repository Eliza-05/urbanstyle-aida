# Guía para Sebastián: empezar las Tareas A y C

Corta y práctica. Todo lo que dice aquí está comprobado contra el código de `develop`.

## 1. Traer los cambios y crear tu rama

```bash
git checkout develop
git pull origin develop                    # trae B, D, el menú y common/
python -m venv .venv                       # Python 3.13
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                                     # debe pasar completo antes de que empieces

git checkout -b feature/task-a-pathfinding   # (para C: feature/task-c-pricing)
python main.py                             # A y C abren como marcador "pendiente"; ESC vuelve al menú
```

Tus PR van **hacia `develop`**, nunca directo a `main`. Haz PR pequeños y frecuentes.

## 2. Contrato de pantalla

Cada tarea es un módulo con una función `run(screen, clock)` que corre su **propio bucle** y **vuelve al menú** cuando el usuario pulsa ESC (o el botón "← Menú"). El menú (`main.py`) la llama así: `module.run(screen, clock)`.

```python
def run(screen, clock):
    view = MiPantalla(size=screen.get_size())
    while not view.done:                                   # view.done = True con ESC o "← Menú"
        dt = clock.tick(config.FPS) / 1000                 # segundos desde el cuadro anterior
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.event.post(pygame.event.Event(pygame.QUIT))   # que el menú también se cierre
                return
            view.handle_event(event)
        view.update(dt)                                    # la velocidad NO debe depender de los FPS
        view.draw(screen)
        pygame.display.flip()
```

- No crees la ventana: la crea `main.py` con `ui.create_window`. Tampoco llames a `pygame.quit()`.
- **Punto de partida: copia [`common/screen_template.py`](../common/screen_template.py)** a `task_a_pathfinding/screen.py` (o `task_c_pricing/screen.py`). Ya trae el contrato, la cabecera, un slider, un botón, un tooltip y ESC. Pruébala con `python -c "from common import screen_template"` o mirando `tests/test_screen_template.py`.
- Ejemplos completos ya hechos: `task_b_forecasting/screen.py` y `task_d_gan/screen.py`.
- Si algo falla al cargar (modelo ausente, etc.), muestra un mensaje en pantalla en vez de romper el menú (así lo hacen B y D).

## 3. Estilo común (`common/ui.py`)

Lenguaje **editorial**, modo oscuro: sin tarjetas con borde ni sombra; las secciones se separan con espacio, tipografía y líneas finas. Un solo acento (`THEME.accent`).

| Qué | Cómo |
|---|---|
| Colores y tamaños | siempre desde `THEME` (`THEME.text`, `THEME.text_muted`, `THEME.accent`, `THEME.size_headline`, `THEME.margin`…), nunca valores sueltos |
| Píxeles | diseña en px **lógicos** (ventana base 1024×640) y convierte con `ui.px(valor)`; la ventana real puede ser 1536×960 en Windows al 150 % |
| Texto | `ui.blit_text(surface, texto, pos, size, color, weight="serif"/"regular"/"medium"/"semibold"/"bold", anchor=...)`; letra mínima 14 |
| Fuentes | DM Serif Display (`weight="serif"`) para titulares y números grandes; Inter para lo demás |
| Eyebrow | `ui.blit_eyebrow(surface, "Tarea A · Rutas", pos, THEME.accent_text)`: etiqueta en MAYÚSCULAS espaciadas |
| Línea fina | `ui.hairline(surface, x1, x2, y)` (horizontal) y `ui.vhairline(surface, x, y1, y2)` |
| Botón de texto | `ui.Button(rect, "Exportar", primary=True)`; `draw(surface)` y `handle_event(event) -> bool` |
| Slider | `ui.Slider(rect, min, max, valor)`; `handle_event(event)` devuelve el valor; `draw(surface, stops=n)` pone marcas |
| Toast | `toast = ui.Toast()`; `toast.show("Guardado")`; `toast.draw(surface)` en cada cuadro |
| Tooltip | **no es una clase de `ui.py`**: se dibuja a mano con `THEME.bg_raised` + `ui.rounded_surface(..., THEME.hairline_color)`. Copia `draw_tooltip` de `common/screen_template.py` |
| Dibujo suave | `ui.aa_polyline`, `ui.aa_line`, `ui.aa_circle`, `ui.dashed_polyline` |
| Aparición suave | `fade = ui.FadeIn()` al abrir; `fade.draw(surface)` al final de `draw` |

Ejemplo mínimo (cabecera + slider + botón dentro de una pantalla):

```python
from common import ui
from common.ui import THEME, px

# __init__: rects en píxeles físicos (ui.px sobre coordenadas lógicas)
self.slider = ui.Slider(pygame.Rect(px(32), px(200), px(232), px(20)), 1, 50, 10)   # nº de clientes
self.run_button = ui.Button(pygame.Rect(px(32), px(260), px(64), px(30)), "Simular", primary=True)

# handle_event
n_clients = round(self.slider.handle_event(event))
if self.run_button.handle_event(event):
    self.start_simulation(n_clients)

# draw
ui.blit_eyebrow(surface, "Tarea A · Rutas de clientes", (px(32), px(22)), THEME.accent_text)
ui.blit_text(surface, "Dónde se aglomera la tienda", (px(32), px(46)), THEME.size_headline, weight="serif")
ui.hairline(surface, px(32), surface.get_width() - px(32), px(104))
self.slider.draw(surface)
self.run_button.draw(surface)
```

El resto de reglas visuales está en el docstring de `common/ui.py` y en la sección "Interfaz común" del README.

## 4. Datos (`common/data.py`) y `pricing_stats()` para la Tarea C

```python
from common import data
df = data.add_segments(data.load_sales())   # 1000 transacciones, 2023-01-01 a 2024-01-01
```

`load_sales()` valida el CSV (columnas, nulos, `Total Amount = Quantity × Price per Unit`). `add_segments` añade `Series` ("Clothing-Female") y `Age Group` ("18-25"…"56-64").

```python
pricing_stats(df, by=("Product Category", "Gender")) -> pandas.DataFrame
```

- `df`: lo que devuelve `load_sales()` (con o sin `add_segments`).
- `by`: columnas que definen el segmento, por ejemplo `("Product Category", "Gender")` (por defecto), `("Series",)` o `("Age Group",)`.
- Devuelve **una fila por segmento**: las columnas de `by` más `n_transactions, units, revenue, price_min, price_mean, price_median, price_max, qty_mean, age_mean`.
- La firma es estable: cambiarla requiere PR revisado por los dos.

Números reales (fila `Clothing`/`Female` de `pricing_stats(df)`):

| n_transactions | units | revenue | price_min | price_mean | price_median | price_max | qty_mean | age_mean |
|---|---|---|---|---|---|---|---|---|
| 174 | 441 | 81 275 | 25 | 179,22 | 50 | 500 | 2,53 | 43,17 |

Para ese `df`, `Price per Unit` solo toma 5 valores: **25, 30, 50, 300 y 500**; por eso la media (~180) está tan lejos de la mediana (50). Por `Age Group`, la media de precio va de 170,84 (36-45) a 191,92 (18-25) y la cantidad media de 2,40 a 2,63 por compra: las diferencias entre segmentos son **pequeñas**, así que los precios óptimos distintos por segmento los tienes que construir en tu modelo de demanda.

⚠️ **El dataset solo registra compras**: da escalas reales (rango de precio, cantidad, edad), **no** cómo reaccionaría un cliente a otro precio. Esa regla de demanda es un supuesto de simulación: vive en `task_c_pricing/demand.py` y se declara así en el informe.

## 5. Reglas del equipo

- **Cada quien sus carpetas.** Tú: `task_a_pathfinding/` y `task_c_pricing/`. No toques `task_b_forecasting/`, `task_d_gan/`, `main.py`.
- **`common/` solo por pull request**, revisado por la otra persona (si necesitas algo nuevo en `ui.py` o `data.py`, abre PR aparte, pequeño).
- **Los modelos se entrenan con el `train.py` de cada tarea** y se guardan en `models/`; la pantalla y `main.py` **solo los cargan**. Nunca se entrena dentro de Pygame.
- **Semillas fijas:** `config.set_seed()` (`SEED = 42`) antes de entrenar o simular.
- **Rutas relativas:** usa `common/config.py` (`config.ROOT`, `config.MODELS_DIR`, `config.OUTPUTS_DIR`), nunca rutas absolutas.
- **Pruebas:** `pytest` debe pasar antes de cada PR. Agrega pruebas en `tests/` (hoy `test_search.py` y `test_env.py` solo comprueban que el módulo importa; reemplázalas por pruebas reales).
- **Tamaño:** checkpoints pequeños (los de B y D pesan 0,02–1,7 MB). No subas datasets descargados ni checkpoints intermedios (`checkpoints/` está en `.gitignore`).
- Salidas en `outputs/task_a/` y `outputs/task_c/`; informe en `report/`.

## 6. Qué espera la rúbrica (enunciado del PDF `docs/UrbanStyle_Project_Plan-2026_2.pdf`)

Pygame es obligatorio, pero **el centro son los algoritmos**. Detalle completo en `docs/UrbanStyle_Plan_de_Trabajo.md` (§5.1 y §5.3).

**Tarea A: rutas y layout (15 %)**
- Obligatorio: tienda como **grid** (p. ej. 8×6); categorías de producto mapeadas a **secciones**; clientes que entran y se mueven hacia las secciones; **BFS y A\*** implementados; **zonas de alto tráfico** visualizadas.
- Pygame: grid y clientes en movimiento, con sliders/botones para cambiar el **algoritmo** o el **número de clientes**.
- Bonus: mapa de calor superpuesto; probar distintos layouts.
- Cuidar: todas las secciones alcanzables; comparar BFS y A\* en las mismas condiciones (largo de ruta, nodos expandidos, tiempo); medir el tráfico de forma objetiva (visitas por celda).

**Tarea C: precios dinámicos con RL (25 %, la de mayor peso)**
- Obligatorio: **entorno Gym** (estado: precio, segmento, ventas recientes; acciones: ajustar el precio; recompensa: ingreso); agente de **stable-baselines3 (DQN o PPO)**; episodios con ingreso medido; **comparación contra precio fijo**.
- Pygame: gráfica de **ingresos en vivo**, sliders para el **precio inicial** y para **reiniciar** el entorno.
- Bonus: varios productos; varios algoritmos de RL.
- Cuidar: la demanda debe bajar al subir el precio (si no, el agente sube el precio al máximo); acciones discretas; validar el entorno con `check_env` antes de entrenar; comparar contra el **mejor** precio fijo, con los mismos escenarios y semillas; la pantalla carga agentes ya entrenados.

**Integración y reporte (10 % + 10 %)**: al menos una visualización por tarea con botones/sliders; código reproducible; informe claro y presentación corta.
