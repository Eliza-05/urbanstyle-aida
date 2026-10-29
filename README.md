# UrbanStyle

Proyecto final del curso AIDA_M (ECI). Aplicación Pygame con un menú principal que da acceso a cuatro tareas de IA para una tienda de moda:

- **A** – Búsqueda de rutas en la tienda (BFS / A*).
- **B** – Pronóstico de ventas (LSTM, PyTorch).
- **C** – Fijación de precios con aprendizaje por refuerzo (Gymnasium + stable-baselines3).
- **D** – Generación de imágenes/afiches con una GAN.

## Instalación

Requiere Python 3.13 (versión con la que se verificó el proyecto). Debe funcionar en CPU, sin GPU.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución y pruebas

```bash
python main.py   # abre el menú principal
pytest           # no necesita el CSV, GPU ni entrenar nada
```

## Datos y documentación

- CSV de ventas: `data/raw/retail_sales_dataset.csv` (nunca se modifica; los derivados van en `data/processed/`).
- Plan del proyecto (Markdown): `docs/`. Guía para empezar A y C: [`docs/guia_para_sebastian.md`](docs/guia_para_sebastian.md).
- Estado: B y D terminadas; A y C abren como marcador "en desarrollo" (Sebastián).

## Tarea A — Rutas de clientes y layout (BFS y A*)

La tienda es una grilla de 16 × 10 celdas con 4 secciones (las 3 categorías del dataset; la ropa separada por género: Ropa mujer, Ropa hombre, Belleza, Electrónica), caja, entrada y salida. Cada cliente es una **transacción real** del CSV: entra, va a la sección de su compra, paga y sale. Cada tramo se planea con **BFS o A\*** (heurística Manhattan) y se cuentan las visitas por celda para encontrar las zonas de alto tráfico. Hay **3 distribuciones** para comparar (bonus): Pasillos, Boutique y Recorrido.

No hay modelo que entrenar. Para reproducir las métricas y figuras:

```bash
python -m task_a_pathfinding.evaluate   # BFS vs. A* en todos los pares y experimento de layouts (~6 s) -> outputs/task_a/
python main.py                          # menú -> "Tarea A"
```

| Archivo | Qué hace |
|---|---|
| `task_a_pathfinding/store.py` | grilla (grafo implícito), secciones, los 3 layouts y su validación (todas las secciones alcanzables) |
| `task_a_pathfinding/search.py` | BFS y A*, con nodos expandidos, tiempo y orden de exploración; `None` si no hay ruta |
| `task_a_pathfinding/simulation.py` | clientes del dataset que recorren la tienda; visitas, ocupación y pico por celda; independiente de los FPS |
| `task_a_pathfinding/evaluate.py` | comparación BFS vs. A* en las mismas condiciones y experimento de layouts |
| `task_a_pathfinding/screen.py` | pantalla: algoritmo, layout, slider de clientes y velocidad, capas Calor / Rutas / Búsqueda, tooltip por celda |

**Resultados** (`outputs/task_a/metrics/`):
- **Mismo largo:** en los 2 976 pares origen-destino, BFS y A* dan rutas del mismo largo el 100 % de las veces.
- **Nodos expandidos:** A* expande entre el 23 % (Boutique) y el 55 % (Recorrido) de los nodos de BFS, y nunca más.
- **Congestión:** el punto más congestionado en los 3 layouts es el pasillo entre la caja y la salida.

Salidas: [`outputs/task_a/README.md`](outputs/task_a/README.md). Decisiones, números y limitaciones: [`docs/notas_informe_tarea_a.md`](docs/notas_informe_tarea_a.md).

## Tarea B — Pronóstico de demanda (LSTM)

El dataset tiene **3 categorías originales** (Beauty, Clothing, Electronics). Con autorización del profesor se trabaja con **6 series = categoría × género** (p. ej. `Clothing-Female`), en unidades vendidas por semana (52 semanas completas, domingo a sábado).

El modelo final ya está en `models/lstm_forecast.pt`, así que no hace falta entrenar para usar la app. Para reproducir todo, ejecuta desde la raíz:

```bash
python -m task_b_forecasting.eda                 # análisis exploratorio  -> outputs/task_b/eda/
python -m task_b_forecasting.tune                # (opcional, ~4.5 min) comparación de modelos con validación
python -m task_b_forecasting.train               # entrena las 6 LSTM (~3 s en CPU) -> models/lstm_forecast.pt
python -m task_b_forecasting.evaluate            # solo evaluar el modelo guardado (sin reentrenar): métricas, gráficas, próxima semana y CSV
python -m task_b_forecasting.evaluate --seeds 5  # lo mismo, y además reentrena con 5 semillas para medir la variabilidad
python -m task_b_forecasting.retrain_check       # (opcional, ~15 s) backtest: ¿reentrenar con train+val?
python main.py                                   # menú -> "Tarea B"
```

| Archivo | Qué hace |
|---|---|
| `common/data.py` | carga y valida el CSV, crea las series y la tabla semanal; `pricing_stats` para la Tarea C |
| `task_b_forecasting/prepare.py` | split cronológico 36/8/8 semanas, MinMax ajustado solo con train, ventanas deslizantes |
| `task_b_forecasting/baselines.py` | MAE, RMSE y líneas base: ingenua, media móvil (k elegido con validación) y media histórica |
| `task_b_forecasting/model.py` | LSTM + capa lineal; guardar y cargar el checkpoint |
| `task_b_forecasting/train.py` | entrenamiento (Adam, MSE, dropout, early stopping, semilla fija) |
| `task_b_forecasting/tune.py` | por serie vs. compartido y elección de hiperparámetros con validación |
| `task_b_forecasting/evaluate.py` | LSTM vs. media móvil, media histórica e ingenua en las mismas semanas de prueba; sesgo, bootstrap y `predictions.csv` |
| `task_b_forecasting/retrain_check.py` | backtest sin tocar prueba para decidir si reentrenar con train+val |
| `task_b_forecasting/screen.py` | pantalla: slider de serie, gráfica real vs. predicho, métricas, próxima semana, botón Exportar CSV |

Salidas organizadas en `outputs/task_b/` (`export/`, `metrics/`, `training/`, `eda/`, `figures/`); qué es cada archivo: [`outputs/task_b/README.md`](outputs/task_b/README.md).

Decisiones, números y limitaciones: [`docs/notas_informe_tarea_b.md`](docs/notas_informe_tarea_b.md).

## Tarea D — Imágenes promocionales con GAN (DCGAN + cGAN)

Una **DCGAN** (la arquitectura del taller W2_DCGAN) genera prendas nuevas de 28×28 a partir de ruido. Una **GAN condicional (cGAN)** (bonus) genera la prenda que se le pide (Camiseta, Pantalón, …, Botín). Con la cGAN se compone un **póster promocional** de UrbanStyle (bonus). Los datos son Fashion-MNIST, que `torchvision` descarga solo en `data/FashionMNIST/`.

Los modelos finales ya están en `models/` (`gan_generator.pt`, `cgan_generator.pt` y la CNN auxiliar de evaluación `fashion_cnn.pt`), así que no hace falta entrenar para usar la app. Para reproducir todo, ejecuta desde la raíz:

```bash
python -m task_d_gan.data              # grilla de imágenes reales (descarga Fashion-MNIST la primera vez)
python -m task_d_gan.classifier        # CNN auxiliar para evaluar (~1,5 min en CPU, 91 % en test)
python -m task_d_gan.train             # entrena DCGAN y cGAN, 20 épocas cada una (~13 min en CPU)
python -m task_d_gan.evaluate          # evalúa los modelos guardados, sin reentrenar: cobertura de clases, control de la cGAN, memorización, FID-CNN (~20 s)
python -m task_d_gan.poster --seed 7   # póster promocional -> outputs/task_d/poster/
python -m task_d_gan.train --explore 30  # (opcional, ~21 min) exploración con la que se eligieron las 20 épocas
python -m task_d_gan.seeds               # (opcional, ~65 min) robustez: DCGAN y cGAN con 5 semillas más
python main.py                         # menú -> "Tarea D"
```

| Archivo | Qué hace |
|---|---|
| `task_d_gan/data.py` | Fashion-MNIST en [-1, 1] (por la Tanh del generador), nombres de las clases en español, grillas |
| `task_d_gan/models.py` | `Generator` y `Discriminator` del taller; `CondGenerator` y `CondDiscriminator` (cGAN); generación determinista por semilla; checkpoints |
| `task_d_gan/train.py` | entrenamiento alterno (`detach` en el paso de D, pérdida no saturante), ruido fijo, grilla por época, GIF, curvas, D(x)/D(G(z)) y FID-CNN por época |
| `task_d_gan/classifier.py` | CNN pequeña que solo sirve para **evaluar** las GAN (no genera nada) |
| `task_d_gan/metrics.py` | FID-CNN, entropía de clases, diversidad por pares y vecino más cercano |
| `task_d_gan/evaluate.py` | evaluación completa -> `outputs/task_d/evaluation/summary.json` y figuras |
| `task_d_gan/seeds.py` | robustez entre semillas: reentrena con las semillas 1–5 (solo para evaluar; los pesos no se suben) y compara con el modelo final (semilla 42, fijada de antemano) |
| `task_d_gan/poster.py` | póster con prendas generadas por la cGAN y la nota visible "Prendas generadas por IA" |
| `task_d_gan/screen.py` | pantalla: botón Generar, slider de semilla, DCGAN/cGAN, prenda, interpolación animada, Crear póster |

**Resultados** (`summary.json`):
- **DCGAN:** cubre las 10 clases (entre 7,7 % y 13,4 % cada una), sin colapso de modos.
- **cGAN:** el 90,9 % de las imágenes es de la clase pedida; la propia CNN acierta el 91,1 % en reales.
- **Memorización:** ninguna muestra está más cerca del entrenamiento que el 1 % más cercano de imágenes reales nuevas, así que no copian.

Salidas en `outputs/task_d/` (`samples/`, `training/`, `evaluation/`, `poster/`, `figures/`); qué es cada archivo: [`outputs/task_d/README.md`](outputs/task_d/README.md). Teoría, decisiones con números, limitaciones y preguntas probables: [`docs/notas_informe_tarea_d.md`](docs/notas_informe_tarea_d.md).

## Interfaz común

`common/ui.py` define el lenguaje visual de toda la app: **editorial** (periodismo de datos + revista de moda), en modo oscuro. Sebastián puede reutilizarlo tal cual en A y C.

- **Principios:** sin tarjetas con borde ni sombra. Las secciones se separan con espacio, jerarquía tipográfica y líneas finas (`ui.hairline`). La gráfica es la protagonista, y hay un solo acento de marca.
- **Tipografía:** **DM Serif Display** (`weight="serif"`, titulares y números grandes) e **Inter** (`regular`, `medium`, `semibold`, `bold`) para el resto. Las dos tienen licencia SIL OFL y están en `assets/fonts/` con su licencia. `ui.blit_eyebrow` hace las etiquetas en mayúsculas espaciadas.
- **Paleta (`THEME`):**
  - fondo `#0F0E0D`; texto `#EFEBE4`, `#B7B0A8` y `#8F8983`;
  - acento `#FF5A36`;
  - modelos: LSTM = acento, media móvil `#B09A74`, media histórica `#7E9C89`, ingenua `#77716B`, barras de ventas `#3D3936`.
- **Nitidez en Windows:** `ui.create_window((1024, 640))` hace el proceso consciente de DPI y crea la ventana en píxeles físicos (1536×960 con escala 150 %). Cada pantalla diseña en píxeles **lógicos** y los convierte con `ui.px(valor)`; los tamaños de letra van en px lógicos (mínimo 14). `URBANSTYLE_SCALE` fuerza una escala (útil para capturas).
- **Dibujo:** texto cacheado con antialias (`ui.blit_text`, `ui.render_text`, `ui.fit_size` para titulares que deben caber), líneas y círculos con antialias (`ui.aa_polyline`, `ui.aa_line`, `ui.aa_circle`, `ui.dashed_polyline`). Lo costoso se pre-renderiza y se cachea.
- **Componentes, con la API de siempre:**
  - `Button`: enlace de texto con subrayado de acento al pasar el mouse; `primary=True` lo pone en el color de acento; con `subtitle` y `badge` es una fila de lista tipográfica.
  - `Slider`: riel de 1 px con marcas.
  - `Toast` (aviso sobrio) y `FadeIn` (aparición suave).
  - `draw_card` y `pill` se mantienen solo por compatibilidad.
  - El tooltip no es un componente de `ui.py`: se dibuja a mano con `THEME.bg_raised` (ejemplo en `common/screen_template.py`).
- **Punto de partida para una pantalla nueva:** [`common/screen_template.py`](common/screen_template.py) (cabecera, zona principal, slider, botón, tooltip y ESC para volver; cópiala a tu carpeta). Contrato con el menú: `run(screen, clock)` corre su propio bucle y vuelve con ESC.
- **Para A y C:** lee [`docs/guia_para_sebastian.md`](docs/guia_para_sebastian.md) (cómo traer los cambios, contrato de pantalla, estilo, `pricing_stats()` y qué pide la rúbrica).

## Responsables

| Carpeta | Responsable |
|---|---|
| `task_a_pathfinding/` | Sebastián |
| `task_b_forecasting/` | Elizabeth |
| `task_c_pricing/` | Sebastián |
| `task_d_gan/` | Elizabeth |
| `main.py`, `common/data.py`, `common/ui.py`, `common/screen_template.py`, `assets/fonts/` | Elizabeth |
| `common/config.py`, `tests/`, `models/`, `outputs/`, `report/`, `docs/` | Compartido |

## Reglas de trabajo

- Cada quien toca solo sus carpetas.
- Los cambios en `common/` van por rama + pull request revisado por la otra persona.
- Los modelos se entrenan con el `train.py` de cada tarea y se guardan en `models/`; `main.py` y las pantallas solo los cargan.
- Siempre rutas relativas (usar `common/config.py`), nunca absolutas.
- Semillas fijas: llamar a `set_seed()` (`SEED = 42`).
