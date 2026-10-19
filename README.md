# UrbanStyle

Proyecto final del curso AIDA_M (ECI). Aplicación Pygame con un menú principal que da acceso a cuatro tareas de IA para una tienda de moda:

- **A** – Búsqueda de rutas en la tienda (BFS / A*).
- **B** – Pronóstico de ventas (LSTM, PyTorch).
- **C** – Fijación de precios con aprendizaje por refuerzo (Gymnasium + stable-baselines3).
- **D** – Generación de imágenes/afiches con una GAN.

## Instalación

Requiere Python 3.11. Debe funcionar en CPU, sin GPU.

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
- Plan del proyecto (Markdown): `docs/`.

## Tarea B — Pronóstico de demanda (LSTM)

El dataset tiene **3 categorías originales** (Beauty, Clothing, Electronics). Con autorización del profesor se trabaja con **6 series = categoría × género** (p. ej. `Clothing-Female`), en unidades vendidas por semana (52 semanas completas, domingo a sábado).

El modelo final ya está en `models/lstm_forecast.pt`, así que no hace falta entrenar para usar la app. Para reproducir todo, ejecuta desde la raíz:

```bash
python -m task_b_forecasting.eda                 # análisis exploratorio  -> outputs/task_b/eda_*
python -m task_b_forecasting.tune                # (opcional, ~4.5 min) comparación de modelos con validación
python -m task_b_forecasting.train               # entrena las 6 LSTM (~3 s en CPU) -> models/lstm_forecast.pt
python -m task_b_forecasting.evaluate --seeds 5  # métricas, gráficas, próxima semana y CSV -> outputs/task_b/
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

Decisiones, números y limitaciones: [`docs/notas_informe_tarea_b.md`](docs/notas_informe_tarea_b.md).

## Responsables

| Carpeta | Responsable |
|---|---|
| `task_a_pathfinding/` | Sebastián |
| `task_b_forecasting/` | Elizabeth |
| `task_c_pricing/` | Sebastián |
| `task_d_gan/` | Elizabeth |
| `main.py`, `common/data.py`, `common/ui.py` | Elizabeth |
| `common/config.py`, `tests/`, `models/`, `outputs/`, `report/`, `docs/` | Compartido |

## Reglas de trabajo

- Cada quien toca solo sus carpetas.
- Los cambios en `common/` van por rama + pull request revisado por la otra persona.
- Los modelos se entrenan con el `train.py` de cada tarea y se guardan en `models/`; `main.py` y las pantallas solo los cargan.
- Siempre rutas relativas (usar `common/config.py`), nunca absolutas.
- Semillas fijas: llamar a `set_seed()` (`SEED = 42`).
