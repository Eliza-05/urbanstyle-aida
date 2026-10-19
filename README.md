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
