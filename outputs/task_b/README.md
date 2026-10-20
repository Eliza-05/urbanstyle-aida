# Salidas de la Tarea B (pronóstico de demanda)

Todo se regenera con los scripts de `task_b_forecasting/` (ver el README principal). Unidades: unidades vendidas por semana.

## export/ — predicciones (bonus: exportación CSV)
El botón "Exportar CSV" de la pantalla (y `python -m task_b_forecasting.evaluate`) genera los dos primeros:

- `predictions.csv`: formato **estándar** (separador `,`, punto decimal, 3 decimales). Es el que se usa con pandas y para reproducir resultados.
- `predictions_excel.csv`: la **misma tabla** para Excel en español: separador `;`, coma decimal, 2 decimales y codificación UTF-8 con BOM (para que Excel reconozca las tildes). Se abre con doble clic y queda en columnas.
- `next_week.csv`: pronóstico de la semana del 2023-12-31 al 2024-01-06 para cada serie y modelo.
- No se genera `.xlsx`: `openpyxl` no está en `requirements.txt` y no se agregó una dependencia solo para eso.

**Columnas de `predictions.csv` / `predictions_excel.csv`** (una fila por semana y serie; unidades vendidas por semana):

| Columna | Qué es |
|---|---|
| `week_start` | Domingo en que empieza la semana (AAAA-MM-DD) |
| `series` | Serie Categoría-Género (p. ej. `Clothing-Female`) |
| `part` | Tramo: `train` (semanas 1–36), `val` (37–44), `test` (45–52, las únicas usadas en las métricas) o `next_week` (semana del 2023-12-31, sin valor real) |
| `actual` | Unidades vendidas realmente |
| `lstm` | Predicción de la LSTM de esa serie |
| `moving_average` | Media móvil de las 8 semanas anteriores |
| `hist_mean_train` | Media histórica: promedio de las semanas 1–36 (solo entrenamiento) |
| `hist_mean_trainval` | Media histórica: promedio de las semanas 1–44 (entrenamiento + validación) |
| `naive` | Línea base ingenua: las ventas de la semana anterior |

**Por qué hay celdas vacías:**
- `hist_mean_train` está vacía en las filas `train`, y `hist_mean_trainval` en las filas `train` y `val`. Esas semanas se usan para calcular el promedio, así que no serían una predicción.
- `actual` está vacía en `next_week`: esa semana aún no se ha observado.
- Las filas empiezan en la **semana 9** (2023-02-26) para todos los modelos: es la primera semana con las 8 semanas previas que necesita la media móvil. Por eso `lstm` y `moving_average` nunca están vacías en el archivo. En la pantalla, la LSTM se dibuja desde la semana 5 (le bastan 4 semanas previas), pero esas semanas 5–8 no van al CSV, para que todos los modelos cubran las mismas filas.

## metrics/ — evaluación en las 8 semanas de prueba
- `metrics.csv`: MAE y RMSE por serie y global de cada modelo.
- `bootstrap.csv`: intervalos de confianza del 95 % (bootstrap pareado) de MAE, RMSE y de la diferencia de cada modelo con la LSTM.
- `residuals.csv`: error medio (predicción − real) por serie: sesgo de cada modelo.
- `metrics_comparison.png`: barras de MAE y RMSE por serie y modelo.

## training/ — entrenamiento y selección del modelo
- `loss_curve.png`: curvas de pérdida (train y validación) de las 6 LSTM, con la época elegida por early stopping.
- `loss_history.csv`: pérdida por época y serie (datos de la curva).
- `tuning_results.csv`: búsqueda de hiperparámetros con validación (modelo por serie vs. compartido).
- `seed_results.csv`: métricas en prueba al reentrenar con 5 semillas (variabilidad).
- `retrain_check.csv`: backtest para decidir si reentrenar con train+val (sin tocar prueba).

## eda/ — análisis exploratorio
- `eda_series.png`: ventas semanales de las 6 series con media móvil y media anual.
- `eda_dispersion.png`: distribución semanal, semanas en cero y autocorrelación por rezago.
- `eda_stats.csv`: estadísticas por serie (media, desviación, CV, ceros, autocorrelación).

## figures/ — figuras para el informe
- `forecast_test.png`: real vs. predicho de las 6 series, con la zona de prueba y la próxima semana.
- `screen_task_b.png`: captura de la pantalla Pygame de la Tarea B a escala real (1536×960); `screen_task_b_75.png` es la misma al 75 %.
- `screen_task_b_global.png`: la pantalla con el selector de la tabla en "Global (6)" (y su versión `_75`).
- `screen_task_b_info.png`: la pantalla con la ayuda del ícono ⓘ visible (y su versión `_75`).
- `screen_task_b_tooltip.png`: la pantalla con el tooltip de una semana de prueba.
- `screen_task_b_antes.png`: captura de la pantalla antes del rediseño visual (para comparar).
- `menu.png` / `menu_hover.png`: menú principal en reposo y con el mouse sobre la Tarea B (y sus versiones `_75`).
