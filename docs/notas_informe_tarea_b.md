# Tarea B — Pronóstico de demanda con LSTM: notas para el informe

> Documento de trabajo con supuestos, decisiones (con números) y resultados reales de la Tarea B.
> Todo número de este documento sale de un script del repositorio; al lado se indica cuál.

## 1. Requisitos (enunciado `UrbanStyle_Project_Plan-2026_2.pdf`)

**Objetivo:** predecir ventas futuras para apoyar decisiones de inventario.

| # | Obligatorio | Dónde se cumple |
|---|---|---|
| 1 | Seleccionar 5–8 categorías | 6 series Categoría × Género (§2) |
| 2 | Preprocesar como series de tiempo semanales | `common/data.py::weekly_series` |
| 3 | Entrenar una LSTM pequeña o un encoder-decoder | `task_b_forecasting/model.py`, `train.py` |
| 4 | Predecir las ventas de la siguiente semana | `evaluate.py` → `forecast_next_week` |
| 5 | Evaluar con MAE y RMSE | `evaluate.py` |
| Pygame | Gráfica del pronóstico, slider para elegir categoría **o** duración del entrenamiento | `screen.py` (slider de serie) |
| Bonus | Línea base de media móvil; exportar predicciones a CSV | `baselines.py`, `outputs/task_b/predictions.csv` |

Rúbrica: 20 % (preparación de series, LSTM funcionando, MAE y RMSE reportados; bonus: comparación con línea base y exportación CSV).
El PDF y `docs/UrbanStyle_Plan_de_Trabajo.md` (§5.2) coinciden en todos los puntos de la Tarea B.

## 2. Datos y series

### 2.1 Autorización de las series

El dataset tiene **solo 3 categorías originales** (Beauty, Clothing, Electronics). Respuesta del profesor:

> "Sí, pueden utilizar el mismo dataset de Kaggle. Para esta tarea pueden trabajar con 5–8 series de demanda, aunque el dataset tenga originalmente solo tres categorías. Pueden construir las series combinando categoría de producto × género, por ejemplo: Clothing-Male, Clothing-Female, Beauty-Male, Beauty-Female, Electronics-Male y Electronics-Female. Deben dejar claramente identificado que las categorías originales son tres y que las seis series resultan de esa segmentación. No es necesario buscar otro dataset."

Por eso se trabaja con **6 series = 3 categorías originales × 2 géneros**: Clothing-Male, Clothing-Female, Beauty-Male, Beauty-Female, Electronics-Male, Electronics-Female. En el informe y en la interfaz se llaman *series* o *segmentos*, no categorías nuevas.

### 2.2 Dataset

`data/raw/retail_sales_dataset.csv`: 1000 transacciones, del domingo 2023-01-01 al lunes 2024-01-01, sin nulos ni duplicados. `load_sales()` valida las 9 columnas, sus tipos y que `Total Amount = Quantity × Price per Unit` en todas las filas.

### 2.3 Decisión: definición de semana → domingo a sábado, 52 semanas completas

| Opción | Semanas completas | Filas que quedan | Filas descartadas | Días descartados |
|---|---|---|---|---|
| **Dom–Sáb** (elegida) | 52 (2023-01-01 a 2023-12-30) | **997** | 3 (6 unidades) | 2023-12-31 y 2024-01-01 |
| Lun–Dom | 52 (2023-01-02 a 2023-12-31) | 995 | 5 (14 unidades) | 2023-01-01 y 2024-01-01 |

Ambas dan 52 semanas, pero Dom–Sáb pierde menos datos porque el CSV empieza un domingo. Las semanas incompletas de los extremos se descartan, porque una semana de 1 o 2 días parecería una caída de ventas que no ocurrió. Comprobación: la suma de las 52 semanas (2508 unidades) es igual a la suma de las 997 filas dentro del rango (prueba `test_integration_real_csv`).

### 2.4 Decisión: pronosticar unidades (Quantity), no ingresos

| Medida | CV semanal promedio de las 6 series | Autocorrelación lag-1 (rango) |
|---|---|---|
| **Unidades** | **0.61** | −0.24 a 0.20 |
| Ingresos | 0.91 | −0.18 a 0.14 |

- **Estabilidad:** los ingresos varían un 50 % más (CV = desviación/media). El precio unitario solo toma 5 valores (25, 30, 50, 300 y 500), y una sola venta de 500 cambia el ingreso de la semana mucho más que el número de prendas.
- **Interpretabilidad:** el objetivo de negocio es el **inventario**, que se planea en unidades.

### 2.5 Hipótesis: semanas sin ventas = 0

- Cobertura de fechas: de 366 días, 21 no tienen ninguna transacción y **nunca hay 2 días seguidos sin datos**. No hay huecos largos que indiquen pérdida de información.
- Semanas en 0 por serie (unidades): Clothing-Male 3, Clothing-Female 1, Beauty-Male 3, Beauty-Female 5, Electronics-Male 1, Electronics-Female 4.
- Con ~3 transacciones por semana y serie, es plausible que una serie no venda nada en una semana. Por eso esas semanas se toman como **0 ventas**. Es una **hipótesis de trabajo**: el dataset no permite distinguir "no hubo ventas" de "no se registró".

### 2.6 `pricing_stats` (para la Tarea C)

Firma estable: `pricing_stats(df, by=("Product Category", "Gender")) -> DataFrame`, con una fila por segmento y las columnas `n_transactions, units, revenue, price_min, price_mean, price_median, price_max, qty_mean, age_mean`. `by` acepta cualquier columna del df o de `add_segments` (`"Series"`, `"Age Group"`). El dataset solo registra compras: da **escalas** reales, no la reacción de los clientes al precio.
