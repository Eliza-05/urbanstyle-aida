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
- Semanas en 0 después del agrupado semanal (de 52), por parte del split:

| Serie | Total | Train (0–35) | Val (36–43) | Prueba (44–51) |
|---|---|---|---|---|
| Clothing-Male | 3 | 1 | 2 | 0 |
| Clothing-Female | 1 | 1 | 0 | 0 |
| Beauty-Male | 3 | 3 | 0 | 0 |
| Beauty-Female | 5 | 3 | 2 | 0 |
| Electronics-Male | 1 | 1 | 0 | 0 |
| Electronics-Female | 4 | 4 | 0 | 0 |

- Con ~3 transacciones por semana y serie, es plausible que una serie no venda nada en una semana. Por eso esas semanas se toman como **0 ventas**. Es una **hipótesis de trabajo**: el dataset no permite distinguir "no hubo ventas" de "no se registró".

### 2.6 `pricing_stats` (para la Tarea C)

Firma estable: `pricing_stats(df, by=("Product Category", "Gender")) -> DataFrame`, con una fila por segmento y las columnas `n_transactions, units, revenue, price_min, price_mean, price_median, price_max, qty_mean, age_mean`. `by` acepta cualquier columna del df o de `add_segments` (`"Series"`, `"Age Group"`). El dataset solo registra compras: da **escalas** reales, no la reacción de los clientes al precio.

## 3. Análisis exploratorio (`python -m task_b_forecasting.eda`)

Salidas: `outputs/task_b/eda_series.png`, `eda_dispersion.png` y `eda_stats.csv`.

| Serie | Unidades totales | Media/sem | Desv. | CV | Máx | Semanas en 0 | ACF lag 1 |
|---|---|---|---|---|---|---|---|
| Clothing-Male | 453 | 8.71 | 5.08 | 0.58 | 19 | 3 | 0.20 |
| Clothing-Female | 441 | 8.48 | 5.09 | 0.60 | 21 | 1 | −0.03 |
| Beauty-Male | 350 | 6.73 | 4.32 | 0.64 | 19 | 3 | 0.03 |
| Beauty-Female | 418 | 8.04 | 4.76 | 0.59 | 20 | 5 | −0.24 |
| Electronics-Male | 407 | 7.83 | 4.59 | 0.59 | 21 | 1 | 0.10 |
| Electronics-Female | 439 | 8.44 | 5.56 | 0.66 | 24 | 4 | 0.04 |

**Conclusiones con números:**
- **Series pequeñas y ruidosas:** de 7 a 9 unidades por semana, unas 3 transacciones, con un CV de ~0.6. Es la limitación aceptada del dataset: 1000 transacciones repartidas en 6 series × 52 semanas.
- **Casi no hay memoria temporal:** con 52 puntos, una autocorrelación menor que ±1.96/√52 ≈ ±0.27 no se distingue de cero. Solo 4 de 24 autocorrelaciones (lags 1–4) quedan fuera de esa banda, con signos mezclados y sin un rezago común entre series. Por puro azar se esperarían ~1.2 de 24.
- **Promediar le gana a copiar la semana anterior:** descriptivamente, sobre todo el año, predecir con la semana anterior da un MAE de 4.75 a 6.29, y predecir con la media anual da 3.44 a 4.46, en las 6 series. Esto anticipa que una media móvil larga será una línea base difícil de superar.
- **Cambios de nivel:** Clothing-Male baja de 10.8 a 6.6 unidades/semana entre el primer y el segundo semestre, y Electronics-Female sube de 6.9 a 10.0. Una media móvil se adapta a estos cambios mejor que la media anual.
- **¿Tiene sentido un LSTM?** El enunciado lo exige y se entrena, pero con este ruido el techo de cualquier modelo es bajo. **Riesgos:** sobreajuste (pocas ventanas por serie) y que el modelo termine aprendiendo solo la media. Mitigaciones evaluadas: un modelo compartido entre las 6 series (6× más ejemplos; en validación resultó peor que uno por serie, §6.2), una red pequeña, dropout, early stopping y una comparación honesta con las líneas base en las mismas semanas.

## 4. Preparación para el modelo (`task_b_forecasting/prepare.py`)

### 4.1 Decisión: split cronológico 36 / 8 / 8 semanas

| Parte | Semanas (índice) | Fechas de inicio | Uso |
|---|---|---|---|
| Entrenamiento | 0–35 (36) | 2023-01-01 a 2023-09-03 | ajustar pesos y escalador |
| Validación | 36–43 (8) | 2023-09-10 a 2023-10-29 | early stopping, elegir k de la media móvil, ventana e hiperparámetros |
| Prueba | 44–51 (8) | 2023-11-05 a 2023-12-24 | solo para el informe final; no se usa para decidir nada |

- **Por qué cronológico:** en una serie de tiempo el modelo solo puede usar el pasado. Si se barajara, entrenaría con semanas de diciembre para "predecir" octubre: eso es fuga de información y daría métricas optimistas que no se repiten en la práctica.
- **Por qué 8 semanas de prueba:** dan 8 × 6 = 48 predicciones para evaluar (unos 2 meses). Con 4 semanas (24 puntos) la métrica variaría demasiado; con 12+12 quedarían solo 28 semanas para entrenar. Con validación del mismo tamaño, 36 semanas de entrenamiento siguen siendo suficientes para ventanas de 4 a 8 semanas.
- **Alternativa descartada:** validación cruzada temporal (rolling origin). Es más robusta, pero no se vio en clase y multiplica el costo; aquí se compensa repitiendo el entrenamiento con 5 semillas.

### 4.2 Escalado MinMax ajustado solo con entrenamiento

Se usa un `MinMaxScaler` (como en los notebooks de clase) **por serie**, ajustado únicamente con las 36 semanas de entrenamiento. Si se ajustara con todo el año, el mínimo y el máximo de prueba se filtrarían al modelo. En validación y prueba puede haber valores fuera de [0, 1]: es correcto, porque el modelo nunca vio esas semanas. Para el checkpoint se guardan `data_min_` y `data_max_`, y `scaler_from_minmax` reconstruye un escalador idéntico.

### 4.3 Ventanas deslizantes

Cada ejemplo usa como entrada las `window` semanas anteriores de una serie y como objetivo la semana siguiente. Así una serie de 52 semanas se convierte en pares (entrada, objetivo). Una ventana de entrenamiento solo usa semanas < 36, tanto en la entrada como en el objetivo. Las ventanas de validación y prueba sí pueden incluir semanas anteriores de otras partes en la **entrada**: es la historia real que se conoce el día del pronóstico. El **objetivo** siempre pertenece a la parte que se evalúa. El tamaño de ventana (4 u 8) se elige con validación (§6).

**Pruebas que demuestran que no hay fuga** (`tests/test_forecasting.py`): `test_scaler_uses_only_train_weeks` (alterar val/test no cambia el escalador), `test_train_windows_never_touch_val_or_test` (ninguna entrada ni objetivo de train viene de una semana ≥ 36), `test_test_windows_target_only_test_weeks`, más pruebas de forma de los tensores y del contenido exacto de una ventana.

## 5. Líneas base (`task_b_forecasting/baselines.py`)

- **Ingenua:** la semana t vende lo mismo que la t−1.
- **Media histórica:** la semana t vende el promedio de una serie sobre un tramo fijo del pasado: solo entrenamiento (semanas 0–35) o entrenamiento + validación (0–43). No tiene parámetros que elegir. Se añadió en la revisión (§13) porque, si la autocorrelación es ~0, el predictor natural es un nivel fijo.
- **Media móvil:** la semana t vende el promedio de las k semanas anteriores. Se usa **un mismo k** para las 6 series, elegido **solo con validación** entre {2, 3, 4, 6, 8, 12}. Con 8 semanas de validación por serie, un k distinto por serie sobreajustaría la propia línea base.

| k | MAE val | RMSE val |
|---|---|---|
| 2 | 5.021 | 5.819 |
| 3 | 4.549 | 5.504 |
| 4 | 4.234 | 5.276 |
| 6 | 4.094 | 5.081 |
| **8** | **3.859** | **4.745** |
| 12 | 3.884 | 4.787 |

→ **k = 8**. El error baja a medida que k crece, lo que confirma el EDA: el ruido es tan grande que promediar más semanas ayuda. MAE y RMSE se verifican a mano (`test_metrics_by_hand`) y contra `sklearn.metrics` (`test_metrics_match_sklearn`).

## 6. Modelo LSTM (`model.py`, `train.py`, `tune.py`)

### 6.1 Arquitectura (la de los notebooks de clase, en PyTorch)

`LSTMForecaster` = `nn.LSTM(1 → hidden)` → último estado oculto → `nn.Dropout` → `nn.Linear(hidden → 1)`. Se entrena con MSE y Adam, con mini-lotes de 32 ventanas barajadas (se barajan las *ventanas*, nunca el orden de las semanas dentro de una ventana). Se usa early stopping sobre la pérdida de validación con paciencia de 30 épocas (máximo 300) y se restauran los pesos de la mejor época. La semilla es fija (`set_seed(42)`) y todo corre en CPU. El dropout va antes de la capa lineal porque con una sola capa LSTM el `dropout` interno de `nn.LSTM` no tiene efecto.

### 6.2 Decisión: una LSTM por serie (A) en vez de una compartida (B)

`python -m task_b_forecasting.tune` (264 s en CPU) compara, **solo con validación** y con 3 semillas por configuración:
- **(A) por serie:** 6 LSTM independientes. Entrada: la serie sola.
- **(B) compartida:** una LSTM para las 6 series, con el one-hot de la serie concatenado a cada paso. Se eligió one-hot y no un *embedding* porque es lo más simple y se vio en clase.

Grilla: ventana {4, 8} × ocultas {2, 4, 8, 16, 32} × dropout {0, 0.2} × lr {0.001, 0.005} = 40 configuraciones por opción, 80 en total (`outputs/task_b/tuning_results.csv`).

| Opción | Mejor configuración | MAE val (media ± desv., 3 semillas) | RMSE val | Mejor época media |
|---|---|---|---|---|
| **A por serie** | w=8, h=16, d=0, lr=0.005 | **3.825 ± 0.064** | 4.508 | 15 |
| B compartida | w=8, h=16, d=0, lr=0.005 | 3.968 ± 0.158 | 4.807 | 2 |

- Las **18 mejores** configuraciones de la grilla son de la opción A; la primera compartida aparece en el puesto 19.
- La compartida casi no aprende: su mejor época de validación es la 0–6 con lr=0.005, y sigue en ~4.0 de MAE con lr=0.001. Una explicación probable es que la autocorrelación es distinta en cada serie (§3), y forzar una sola dinámica mezcla patrones que se contradicen.
- **Consecuencia:** el checkpoint guarda 6 conjuntos de pesos en un solo archivo.

### 6.3 Decisión: configuración final w=4, h=8, dropout=0.2, lr=0.005

**Regla de selección**, fijada antes de ver los resultados, con dos partes:
1. **Parsimonia:** entre las configuraciones a menos de una desviación estándar de la mejor (3.825 + 0.064 = 3.889), se elige la red más pequeña.
2. **Admisibilidad:** el modelo debe **entrenar de verdad** (el early stopping no puede quedarse en las primeras épocas) y ser **estable entre semillas**.

| Configuración (A, w=4, lr=0.005) | MAE val | Mejor época media | Parámetros por LSTM | ¿Cumple? |
|---|---|---|---|---|
| h=2, d=0 | 3.855 ± 0.115 | 50 (seeds 0–2), pero 0–31 con seeds 42–46 | 43 | Parsimonia sí; admisibilidad **no** |
| h=4, d=0 / d=0.2 | 3.899 / 3.949 | 27 / 31 | 117 | No: fuera de la banda |
| **h=8, d=0.2 (elegida)** | **3.833 ± 0.036** | 54 | **361** | Sí |
| w=8, h=16, d=0 (mejor MAE) | 3.825 ± 0.064 | 15 | 1 233 | Sí, pero más grande |

**Por qué se descartó h=2** aunque cae dentro de la banda:
- Con las semillas 42–46, el early stopping se quedó en la **época 0** en 3 de las 6 series (semilla 42), y la mejor época media saltó de 0.7 a 31 según la semilla. Es una red que casi no se entrena: su buen MAE de validación sale de dónde cae la inicialización tras unos pocos pasos, no de aprender de los datos.
- Su variación entre semillas es 3× mayor (0.115 contra 0.036).
- Transparencia: con h=2 llegaron a calcularse métricas de prueba (MAE 3.358 ± 0.082 en 5 semillas, un poco mejor que h=8). La decisión de descartarlo se tomó por la dinámica de entrenamiento y la estabilidad, no por esos números; de hecho, se descartó la opción con mejor prueba.

**Resto de la configuración:**
- **Ventana de 4 semanas (~1 mes):** la de 8 no mejora fuera del ruido, y con 4 cada serie tiene 32 ventanas de entrenamiento en vez de 28.
- **dropout 0.2:** regulariza, y en la grilla no empeora el resultado.

### 6.4 Entrenamiento final (`python -m task_b_forecasting.train`)

- **Tiempo:** ~2.6 s en CPU para las 6 LSTM.
- **Mejor época por serie:** 15, 59, 26, 16, 26 y 24, con early stopping a las 46–90 épocas.
- **¿Aprende?** Sí:
  - la pérdida de entrenamiento baja de 0.21–0.38 (época 0) a 0.04–0.07;
  - la de validación baja de 0.17–0.29 a 0.03–0.06 en la mejor época;
  - el early stopping restaura esos pesos (`best_state`), y la última época tiene igual o peor validación (p. ej. Clothing-Male 0.054 → 0.070).
- **Curvas** (`outputs/task_b/loss_curve.png`): la pérdida de validación baja hasta la mejor época y luego queda plana o sube un poco. Es una señal leve de sobreajuste que el early stopping corta. **No hay subajuste grave:** el error de entrenamiento se aplana porque lo que queda es ruido. La validación queda por debajo de entrenamiento porque el dropout solo actúa al entrenar y porque la pérdida de entrenamiento es un promedio durante la época.
- **Tamaño de la red vs. datos:** 361 parámetros por LSTM para 32 ventanas de 4 semanas (128 valores de entrada). Nominalmente hay más parámetros que ejemplos, así que la red **podría** memorizar. Que no lo haga se debe a tres frenos: el early stopping corta en la época 15–59, antes de memorizar; el dropout; y que sus predicciones casi no varían (desviación de 0.3–0.5 unidades, §7.3), es decir, no está copiando el ruido de entrenamiento. Una red más chica (h=2) no fue mejor (§6.3).
- **Checkpoint** `models/lstm_forecast.pt` (21 KB): pesos de las 6 LSTM, `scaler_min`/`scaler_max`, ventana, nombres de las series, hiperparámetros, límites del split, k de la media móvil, semilla y mejores épocas. Al recargarlo, las predicciones son **idénticas** (comprobado en `train.py` y en `tests/`).
- **Reproducibilidad:** volver a entrenar con la semilla 42 produce pesos **idénticos al bit** (comprobado en el repo y en un clon limpio).

## 7. Resultados (`python -m task_b_forecasting.evaluate --seeds 5`)

Prueba = las últimas 8 semanas (2023-11-05 a 2023-12-24) × 6 series = **48 predicciones, las mismas filas para todos los modelos**. Métricas en **unidades por semana** (con el escalado ya invertido). Las métricas globales se recalcularon con `sklearn.metrics` y coinciden (`verify_with_sklearn` = True).

**Líneas base:**
- **Media móvil k=8:** promedio de las 8 semanas anteriores.
- **Media histórica (train):** promedio de las semanas 0–35 de la serie, un nivel fijo.
- **Media histórica (train+val):** promedio de las semanas 0–43. Usa la misma información que tenían los demás modelos al empezar la prueba.
- **Ingenua:** la semana anterior.

### 7.1 Checkpoint final (semilla 42), por serie

MAE / RMSE:

| Serie | LSTM | Media móvil k=8 | Media hist. (train) | Media hist. (train+val) | Ingenua |
|---|---|---|---|---|---|
| Clothing-Male | 3.49 / 4.00 | **2.97** / 4.00 | 3.60 / 4.72 | 3.26 / 4.37 | 3.50 / 4.44 |
| Clothing-Female | 3.95 / 4.23 | 4.06 / 4.37 | **3.88** / 4.36 | **3.88** / 4.32 | 5.88 / 6.45 |
| Beauty-Male | 2.45 / 3.32 | 2.59 / 3.82 | **2.14 / 2.70** | 2.22 / 2.78 | 4.00 / 4.74 |
| Beauty-Female | 2.93 / 3.54 | 2.31 / 2.65 | **2.31 / 2.59** | 2.40 / 2.71 | 3.50 / 3.84 |
| Electronics-Male | 3.70 / 4.32 | 3.69 / 4.61 | 3.76 / 4.25 | **3.68 / 4.21** | 5.38 / 6.03 |
| Electronics-Female | **3.89 / 5.08** | 4.42 / 5.84 | 4.19 / 5.61 | 4.00 / 5.40 | 6.50 / 7.19 |
| **GLOBAL** | 3.401 / 4.121 | 3.341 / 4.320 | 3.312 / 4.180 | **3.240 / 4.077** | 4.792 / 5.579 |

### 7.2 Resultado global con variabilidad entre semillas e intervalos de confianza

| Modelo | MAE | RMSE | IC 95 % MAE | IC 95 % RMSE | IC 95 % de (modelo − LSTM) en MAE | en RMSE |
|---|---|---|---|---|---|---|
| **LSTM** (5 semillas, media ± desv.) | **3.430 ± 0.039** | **4.164 ± 0.051** | [2.77, 4.08]* | [3.46, 4.77]* | — | — |
| Media móvil k=8 | 3.341 | 4.320 | [2.61, 4.13] | [3.50, 5.11] | [−0.35, +0.22] | [−0.16, +0.50] |
| Media histórica (train) | 3.312 | 4.180 | [2.63, 4.07] | [3.34, 5.00] | [−0.58, +0.41] | [−0.51, +0.61] |
| Media histórica (train+val) | 3.240 | 4.077 | [2.57, 3.97] | [3.29, 4.84] | [−0.56, +0.24] | [−0.50, +0.40] |
| Ingenua | 4.792 | 5.579 | [4.00, 5.63] | [4.77, 6.33] | **[+0.56, +2.23]** | **[+0.71, +2.18]** |
| *Referencia inalcanzable:* oráculo con la media real de prueba de cada serie | 3.151 | 3.845 | | | | |

\* Intervalos del checkpoint de semilla 42 (MAE 3.401, RMSE 4.121). Fuente: `outputs/task_b/bootstrap.csv` y `seed_results.csv`.

**Método de significancia: bootstrap pareado** (`paired_bootstrap`, 10 000 repeticiones, semilla fija).
- En cada repetición se eligen al azar, con reemplazo, 48 de las 48 filas (semana, serie), **las mismas para todos los modelos**, y se recalculan MAE, RMSE y la diferencia con la LSTM.
- Si el intervalo del 95 % de la diferencia contiene el 0, la diferencia no se distingue del azar con 48 predicciones.
- Limitación: el método trata las filas como independientes. Como la autocorrelación es ~0 (§3), es una aproximación razonable.

**Conclusión de significancia:**
- **LSTM, media móvil y las dos medias históricas son estadísticamente indistinguibles:** todos los intervalos de diferencia contienen el 0. Las diferencias de 0.06–0.16 unidades/semana son ruido.
- **Las cuatro le ganan con claridad a la ingenua**, con intervalos que no tocan el 0.
- La variación de la LSTM entre semillas (desviación de 0.04 en MAE) es mucho menor que la incertidumbre por tener solo 48 predicciones (intervalos de ±0.65).

### 7.3 Interpretación: por qué el resultado es el esperado

El argumento, paso a paso y con números:
1. **La autocorrelación semanal es ~0:** solo 4 de 24 coeficientes (lags 1–4) salen de la banda de ruido ±0.27, con signos mezclados (§3). Las ventas de una semana casi no dicen nada de la siguiente.
2. **Entonces cada serie es casi ruido alrededor de un nivel.** Si el pasado no informa el futuro, el mejor predictor posible, en el sentido de minimizar el error cuadrático, es ese **nivel**: el promedio esperado de la serie.
3. **Ese predictor tiene un piso de error que nadie puede bajar.** Aun un oráculo que conociera de antemano el nivel real de cada serie en las 8 semanas de prueba lograría MAE 3.151 y RMSE 3.845. Ese error es el ruido propio de las ventas.
4. **La LSTM aprendió justamente ese nivel, sin sobreajustar el ruido:**
   - sus predicciones casi no varían de una semana a otra (desviación de 0.3–0.5 unidades, contra 4.3–5.2 de las ventas reales y 1.1–2.7 de la media móvil);
   - queda a solo **0.25 unidades de MAE y 0.28 de RMSE** del oráculo inalcanzable;
   - si hubiera memorizado el ruido de entrenamiento, sus predicciones oscilarían y su error en prueba sería mucho mayor, como el de la ingenua (4.79).
5. **Por eso la LSTM, la media móvil y la media histórica empatan:** las tres estiman el mismo nivel de distintas formas. La media móvil reacciona a las últimas 8 semanas, la media histórica fija un promedio largo y la LSTM aprende un nivel con la validación como guía.

**LSTM vs. media histórica, con honestidad:**
- La media histórica (train+val) tiene el menor error de todos: MAE 3.240 y RMSE 4.077, contra 3.430 ± 0.039 y 4.164 ± 0.051 de la LSTM.
- La diferencia **no es significativa** (intervalos [−0.56, +0.24] y [−0.50, +0.40]).
- **Conclusión práctica:** en este dataset, un promedio simple rinde igual que la LSTM. La LSTM cumple el requisito del enunciado y aprende lo correcto, pero no aporta precisión adicional, porque no hay dinámica temporal que explotar. Si se requiriera un modelo en producción con estos datos, el promedio sería la opción más económica.

**Residuos en prueba** (error medio = predicción − real; `outputs/task_b/residuals.csv`):

| Serie | LSTM | Media móvil | Media hist. (train+val) |
|---|---|---|---|
| Clothing-Male | −1.06 | +0.66 | +2.02 |
| Clothing-Female | +0.63 | +0.53 | +1.31 |
| Beauty-Male | +2.03 | +2.38 | +0.86 |
| Beauty-Female | −2.45 | −1.50 | −0.99 |
| Electronics-Male | +0.81 | +1.66 | −0.35 |
| Electronics-Female | −0.75 | −0.89 | −1.84 |
| **GLOBAL** | **−0.13** | +0.47 | +0.17 |

- **Sesgo global:** el de la LSTM es prácticamente 0 (−0.13), así que no sobreestima ni subestima de forma sistemática.
- **Sesgos por serie:** sí los hay. La LSTM sobreestima Beauty-Male en 2.0 y subestima Beauty-Female en 2.4. **La media móvil comete el mismo error en el mismo sentido** (+2.4 y −1.5): en esas series, el nivel de noviembre–diciembre cambió respecto al de los meses anteriores (Beauty-Male bajó, Beauty-Female subió). Ningún modelo basado solo en el pasado puede anticipar un cambio de nivel sin causa visible en los datos.
- **Sentido de los sesgos:** la LSTM sigue más el nivel reciente (validación, sep–oct) que la media histórica. Por eso sus sesgos difieren, por ejemplo en Clothing-Male, que bajó durante el año: la LSTM −1.06 y la media histórica +2.02.

**Por serie:** en MAE, la LSTM gana a la media móvil en 23 de las 48 filas, prácticamente la mitad. En el agregado por serie cada uno gana en 3 de 6 (MAE).

**Mejoras que se probaron (todas con validación):**
- ventana 4/8, ocultas 2/4/8/16/32, dropout 0/0.2, lr 0.001/0.005, y modelo por serie vs. compartido (§6);
- reentrenamiento con train+val (§7.4).

Ninguna se separa del resto más allá del ruido. Seguir ajustando con 8 semanas de validación sería ajustar ruido.

### 7.4 Pronóstico de la próxima semana y decisión train vs. train+val

**Qué es exactamente la "próxima semana":** la semana del **domingo 2023-12-31 al sábado 2024-01-06**, inmediatamente posterior a la última semana completa del dataset (2023-12-24 a 2023-12-30).
- La entrada de la LSTM son las **últimas 4 semanas observadas** (las que empiezan el 2023-12-03, 10, 17 y 24); la media móvil usa las últimas 8. Lo comprueba `test_forecast_next_week_is_the_week_after_the_data`.
- El CSV tiene 3 transacciones del 31-dic y 1-ene, que están fuera de las 52 semanas completas (§2.3). Esa semana está incompleta en los datos, así que no hay valor real con qué comparar.

**¿Reentrenar con train+val para el pronóstico final?** Se decidió con un backtest que **no toca prueba** (`python -m task_b_forecasting.retrain_check`, `outputs/task_b/retrain_check.csv`). Se repite la situación 8 semanas antes y se predicen las semanas 36–43, que ninguno de los dos modelos vio:
- **A (como el modelo actual):** entrena con las semanas 0–27 y hace early stopping con 28–35.
- **B (reentrenado):** entrena con 0–35 durante las épocas que eligió A, con el escalador ajustado a 0–35.

| 5 semillas | MAE (media ± desv.) | RMSE (media ± desv.) |
|---|---|---|
| A: solo train | 4.017 ± 0.061 | **4.925 ± 0.075** |
| B: reentrenado con train+val | **3.990 ± 0.055** | 5.046 ± 0.076 |

**Decisión: no se reentrena.**
- B baja el MAE en solo 0.027 (en las 5 semillas), menos que la variación entre semillas, y **sube** el RMSE en 0.12.
- No hay beneficio claro, y mantener el modelo entrenado solo con train tiene una ventaja importante: **el modelo que pronostica la próxima semana es exactamente el que se evaluó** en las 8 semanas de prueba. Sus métricas siguen siendo una estimación honesta de su error.
- El pronóstico igual usa los datos más recientes como **entrada**: las 4 últimas semanas.

| Serie | LSTM | Media móvil k=8 | Media hist. (train+val) | Ingenua |
|---|---|---|---|---|
| Clothing-Male | 5.16 | 7.00 | 9.02 | 1 |
| Clothing-Female | 8.15 | 7.38 | 8.68 | 11 |
| Beauty-Male | 7.68 | 6.00 | 6.86 | 6 |
| Beauty-Female | 6.78 | 8.88 | 7.89 | 13 |
| Electronics-Male | 8.32 | 8.12 | 7.77 | 2 |
| Electronics-Female | 9.96 | 10.00 | 8.16 | 12 |

Para inventario, con un MAE de ~3.4 unidades sobre niveles de 6 a 10, conviene presentar el pronóstico como un **rango** (± el MAE) y no como un número exacto. La pantalla lo muestra así.

### 7.5 Exportación (bonus)

`outputs/task_b/predictions.csv` tiene 270 filas.
- **Columnas:** `week_start, series, part, actual, lstm, moving_average, hist_mean_train, hist_mean_trainval, naive`.
- **`part`** ∈ {train (168), val (48), test (48), next_week (6)}.
- **`actual`** está vacío solo en `next_week`.
- **Las medias históricas** valen NaN en las semanas usadas para calcularlas: `hist_mean_train` en train, y `hist_mean_trainval` en train y val. Ahí no serían una predicción.
- **Métricas:** usan solo `test`. La pantalla tiene un botón que regenera este archivo.

## 8. Pantalla Pygame (`task_b_forecasting/screen.py`)

- **Control del enunciado:** un **slider para elegir la serie** (6 posiciones; también funcionan las flechas ← →). No se agregó el slider de duración de entrenamiento: el enunciado pide categoría **o** duración, y ese control exigiría cargar modelos distintos entrenados con cada duración.
- **Gráfica:** ventas reales de las 52 semanas, predicción de la LSTM (línea azul), media móvil k=8 (línea naranja punteada), zona de prueba sombreada y punto de la próxima semana.
- **Panel derecho:**
  - MAE y RMSE en prueba de LSTM, media móvil, media histórica (train+val) e ingenua, para la serie elegida y global.
  - Pronóstico de la próxima semana, LSTM ± su MAE y media móvil.
- **Botones y textos:**
  - Botón **Exportar CSV**, que regenera `outputs/task_b/predictions.csv` y lo confirma en pantalla.
  - Botón **Volver** o tecla ESC.
  - Texto fijo: "3 categorías originales × género = 6 series".
- **Solo carga el modelo:** carga `models/lstm_forecast.pt` y nunca entrena. torch se importa solo al entrar a la pantalla, para que el menú abra rápido, y los resultados quedan en caché. Si falta el checkpoint, muestra el comando para entrenarlo en vez de fallar.
- **Verificación headless** (`SDL_VIDEODRIVER=dummy`):
  - el clic en el slider cambia a Beauty-Female y la flecha derecha a Electronics-Male;
  - la exportación muestra el mensaje y ESC cierra la pantalla;
  - desde `main.py`, clic en "Tarea B" → slider → ESC → cerrar ventana termina limpio;
  - las capturas de las 6 series y de la pantalla de error se revisaron a ojo.
  - Captura: `outputs/task_b/screen_task_b.png`.

## 9. Fundamentos teóricos (para la sustentación)

**¿Por qué LSTM y no una RNN simple?** Una RNN simple pasa su estado de un paso al siguiente multiplicándolo una y otra vez por los mismos pesos. Al entrenar, el gradiente se multiplica igual hacia atrás y se encoge (**desvanecimiento del gradiente**): la red deja de aprender de lo que pasó hace varios pasos. La LSTM añade una **celda de memoria** controlada por tres **compuertas**. La de **olvido** decide qué se borra, la de **entrada** qué se guarda nuevo y la de **salida** qué se muestra. Como la celda se actualiza sumando y no solo multiplicando, el gradiente fluye mejor y la red conserva memoria de más largo plazo. La GRU hace algo parecido con dos compuertas. Se eligió LSTM porque es lo que pide el enunciado.

**Ventana deslizante.** Una red no "ve" una serie entera de una vez; aprende de ejemplos (entrada → objetivo). Una ventana de 4 semanas toma las semanas 1–4 como entrada y la 5 como objetivo; luego se corre una posición (2–5 → 6), y así sucesivamente. Con 36 semanas de entrenamiento salen 32 ejemplos por serie.

**Split cronológico y escalador solo con train.** En la vida real se pronostica el futuro con el pasado. Si se barajan las semanas, el modelo se entrena con información "del futuro" y las métricas salen mejores de lo que serán en la práctica: eso es **fuga de información**. Por lo mismo, el MinMax se ajusta solo con entrenamiento. Si usara el mínimo y el máximo de todo el año, el modelo sabría de antemano el rango de las semanas de prueba.

**MAE vs. RMSE.** El MAE es el error promedio en unidades ("me equivoco en ~3.4 prendas por semana"). El RMSE eleva los errores al cuadrado antes de promediar, así que **castiga más los errores grandes**: dos modelos con el mismo MAE pueden tener RMSE distinto si uno comete fallos grandes ocasionales. Siempre RMSE ≥ MAE. Se reportan en **unidades reales**, invirtiendo el escalado, porque un error de 0.05 en escala [0, 1] no significa nada para quien compra inventario.

**¿Por qué la media móvil y la media histórica son las líneas base correctas?** Son lo que haría una tienda sin modelos: "pido lo que vendí en promedio últimamente" o "lo que vendo en promedio". Son simples, no se entrenan y son difíciles de batir cuando las series son ruidosas. Un modelo complejo solo vale la pena si las supera. **Si empatan o ganan**, significa que en estos datos no hay patrones temporales que la LSTM pueda explotar más allá del nivel promedio. Es un resultado válido y útil (no hace falta un modelo costoso), no un fracaso del método.

**Red pequeña y control del sobreajuste.** Con 32 ejemplos por serie, una red grande memorizaría el ruido de entrenamiento (**sobreajuste**: baja la pérdida de entrenamiento y sube la de validación). Se controla con tres medidas:
- **pocas unidades ocultas:** 8, para 361 parámetros por LSTM;
- **dropout 0.2:** apaga neuronas al azar durante el entrenamiento para que la red no dependa de ninguna;
- **early stopping:** se detiene cuando la pérdida de validación deja de mejorar durante 30 épocas y se recuperan los pesos de la mejor época.

**LSTM vs. encoder-decoder.** Un encoder-decoder usa una LSTM que resume la entrada en un estado (encoder) y otra que genera una **secuencia** de salida paso a paso (decoder). Se usa en traducción o para pronosticar varias semanas a la vez. El enunciado pide predecir **la siguiente semana**, es decir, una sola salida. Por eso basta una LSTM con una capa lineal al final: es más simple, tiene menos parámetros (importante con tan pocos datos) y es igual a los ejemplos de clase.

## 10. Supuestos y limitaciones

- **Las 6 series no son categorías nuevas:** salen de cruzar las 3 categorías originales con el género, con autorización del profesor.
- **Semanas sin ventas = 0:** es una hipótesis (§2.5).
- **Series cortas y ruidosas:** 52 semanas de ~3 transacciones; la autocorrelación casi nula (§3) limita lo que cualquier modelo puede aprender. Es una limitación del dataset, no del método.
- **Prueba pequeña:** 48 predicciones; el bootstrap da intervalos de ±0.65 en MAE (§7.2). Diferencias entre modelos menores que eso no son concluyentes.
- **Cambios de nivel no anticipables:** en noviembre–diciembre algunas series cambiaron de nivel (Beauty-Male bajó, Beauty-Female subió), y todos los modelos se equivocan en el mismo sentido (§7.3).
- **Sin factores externos:** el dataset no trae promociones, festivos ni precios de la competencia. No hay estacionalidad anual medible con un solo año de datos.
- **Transparencia sobre prueba:**
  - durante el desarrollo se imprimieron las métricas de prueba de todos los k candidatos de la media móvil y de la configuración h=2;
  - k se eligió automáticamente con validación, y h=2 se descartó por su dinámica de entrenamiento (§6.3);
  - ninguna decisión se tomó para mejorar el resultado en prueba.
- **Python 3.13:** todo se ejecutó y verificó en Python 3.13.3, porque en la máquina de desarrollo no hay 3.11; falta verificarlo en 3.11.

## 11. Pruebas (`pytest`)

- `tests/test_data.py`: 7 pruebas. Series y rangos de edad, semana Dom/Lun, tabla semanal con ceros y totales, ingresos, `pricing_stats`, validación de columnas e integración con el CSV real (se salta si no hay CSV).
- `tests/test_forecasting.py`: 25 pruebas.
  - split cronológico y escalador solo con train (no-fuga);
  - formas y contenido de las ventanas, y que ninguna ventana de train toque val/test;
  - MAE y RMSE a mano y contra sklearn, líneas base a mano (ingenua, media móvil, media histórica) y `choose_k`;
  - salida del modelo, checkpoint que se recarga con predicciones idénticas y desescalado por serie;
  - alineación de **todos** los modelos en las mismas filas de prueba y pronóstico de la próxima semana;
  - sesgo de residuos y bootstrap pareado con datos sintéticos de resultado conocido;
  - checkpoint real + pantalla headless (se salta si no hay CSV o checkpoint).
- **Ninguna prueba entrena.** Suite completa: 32 pruebas en ~5 s, sobre todo por la importación de torch. Sin el CSV ni el checkpoint: 30 pasan y 2 se saltan.

## 12. Auditoría final de la Tarea B (enunciado + rúbrica + checklist §11 del plan)

| Requisito | Tipo | Estado | Evidencia |
|---|---|---|---|
| Seleccionar 5–8 categorías | Obligatorio | ✅ | 6 series Categoría × Género con autorización del profesor (§2.1); aviso en la pantalla |
| Preprocesar como series semanales | Obligatorio · rúbrica "time-series prep" | ✅ | `weekly_series`: 52 semanas Dom–Sáb, suma = 2508 unidades = filas del CSV; hipótesis de ceros (§2.5) |
| LSTM pequeña o encoder-decoder | Obligatorio · rúbrica "LSTM model runs" | ✅ | 6 LSTM de 361 parámetros, `models/lstm_forecast.pt`, entrenamiento de ~3 s en CPU, pesos reproducibles al bit |
| Predecir la siguiente semana | Obligatorio | ✅ | `next_week.csv`: semana del 2023-12-31 con las 4 últimas semanas observadas como entrada (§7.4) |
| Evaluar con MAE y RMSE | Obligatorio · rúbrica "MAE & RMSE reported" | ✅ | `metrics.csv` por serie y global en unidades reales; coincide con sklearn |
| Pygame: gráfica del pronóstico + slider de categoría o duración | Obligatorio | ✅ | `screen.py` con slider de serie; captura y prueba headless |
| Línea base de media móvil | Bonus · rúbrica "baseline comparison" | ✅ | k=8 elegido con validación; además media histórica e ingenua; mismas 48 filas; bootstrap pareado |
| Exportar predicciones a CSV | Bonus · rúbrica "CSV export" | ✅ | `predictions.csv` (evaluate.py y botón de la pantalla) |
| División cronológica sin fuga | Checklist §11 | ✅ | §4 y pruebas de no-fuga |
| Pruebas pasan; evidencias guardadas | Checklist §11 | ✅ | 32 pruebas; gráficas y tablas en `outputs/task_b/` |
| Python 3.11 | Plan §7 (sugerencia) | ⚠️ | verificado en 3.13.3; falta 3.11 |

## 13. Revisión como experto en ML: hallazgos y correcciones

| # | Hallazgo | Corrección |
|---|---|---|
| 1 | No había una línea base de **nivel fijo**, que es la natural cuando la autocorrelación es ~0 | Media histórica (train y train+val), evaluada en las mismas 48 filas |
| 2 | Las diferencias entre modelos se reportaban sin medir la incertidumbre por tener solo 48 predicciones | Bootstrap pareado con IC 95 %: LSTM, media móvil y medias históricas son indistinguibles; la ingenua es peor |
| 3 | No se analizaba el sesgo de los residuos | `residuals.csv`: sesgo global de la LSTM −0.13; los sesgos por serie los comparten la media móvil (cambios de nivel) |
| 4 | Faltaba justificar el tamaño de la red frente a la cantidad de datos | Grilla ampliada a h=2 y h=4; regla de admisibilidad; explicación de 361 parámetros vs. 32 ventanas (§6.3–6.4) |
| 5 | La decisión train vs. train+val para el pronóstico final no estaba evaluada | Backtest sin tocar prueba (`retrain_check.py`): no se reentrena (§7.4) |
| 6 | El ciclo de entrenamiento no permitía entrenar sin validación (necesario para el backtest) | `train_model` acepta `X_va=None` (épocas fijas); el paso de época se extrajo en `_train_one_epoch` para no duplicar código |
| 7 | La conclusión "LSTM gana en RMSE" sonaba a ventaja | Reescrita: la diferencia no es significativa y la media histórica tiene el menor error (§7.3) |

Se comprobaron sin hallar errores:
- el agrupado semanal (suma 2508 = filas, la semana 2023-11-05 de Clothing-Male recalculada desde las transacciones: 10 = 10);
- las ventanas a mano (Clothing-Male objetivo 2023-11-05 ← semanas 2023-10-08 a 10-29; Beauty-Female 2023-09-10 ← 08-13 a 09-03; Electronics-Female 2023-12-24 ← 11-26 a 12-17);
- el desescalado (salida cruda 0.3012 × rango 19 + mínimo 0 = 5.723 unidades);
- el escalador solo con train;
- el uso de validación solo para k, hiperparámetros y early stopping;
- la reproducibilidad al bit.

## 14. Preguntas probables del profesor

**1. ¿Por qué LSTM y no una RNN simple?**
Por el desvanecimiento del gradiente: en una RNN simple el gradiente se encoge al retroceder en el tiempo y la red olvida lo lejano. La LSTM tiene una celda de memoria con compuertas de olvido, entrada y salida que deja pasar el gradiente y conserva memoria más larga.

**2. ¿Por qué el split es cronológico y no aleatorio?**
Porque en la práctica se pronostica el futuro con el pasado. Barajar dejaría al modelo entrenarse con semanas posteriores a las que predice (fuga de información) y daría métricas irreales. Usamos 36 semanas para entrenar, 8 para validar y las 8 últimas para probar.

**3. ¿Cómo evitaron la fuga de información?**
- El escalador MinMax se ajusta solo con las 36 semanas de entrenamiento.
- Las ventanas de entrenamiento no tocan ninguna semana posterior.
- La validación solo se usa para elegir k, los hiperparámetros y el early stopping.
- La prueba solo se usa para el reporte final.
- Hay pruebas automáticas que lo verifican.

**4. ¿Por qué una LSTM por serie y no una para todas?**
Lo medimos en validación: la mejor por serie da MAE 3.825 y la mejor compartida 3.968, y las 18 mejores configuraciones fueron por serie. La compartida dejaba de mejorar casi desde la primera época: cada serie tiene un comportamiento distinto y mezclarlas confunde a la red.

**5. ¿Por qué la LSTM no supera claramente a la media móvil?**
Porque las ventas de una semana casi no predicen las de la siguiente (autocorrelación ~0). Sin patrón temporal, lo mejor que se puede predecir es el nivel promedio, y eso es lo que aprendieron la LSTM, la media móvil y la media histórica. Las tres quedan cerca del piso de error que tendría incluso quien conociera el nivel real de antemano (MAE 3.15), y el bootstrap muestra que sus diferencias son ruido.

**6. Entonces, ¿la LSTM sirve para algo aquí?**
Cumple el requisito y aprendió lo correcto (el nivel), sin sobreajustar el ruido; le gana con claridad a la predicción ingenua (MAE 3.40 contra 4.79). Pero con este dataset no aporta precisión extra sobre un promedio simple, y lo reportamos así.

**7. ¿Qué significan el MAE y el RMSE en este caso?**
Un MAE de 3.4 significa que, en promedio, el pronóstico se equivoca en unas 3.4 unidades por semana y serie, sobre ventas típicas de 6 a 10. El RMSE (4.1) es mayor porque pesa más los errores grandes. Que la LSTM tenga un RMSE relativamente bajo indica que no comete fallos extremos.

**8. ¿Cómo controlaron el sobreajuste con tan pocos datos?**
Con una red pequeña (8 unidades), dropout de 0.2 y early stopping con paciencia de 30 épocas, que restaura los pesos de la mejor época de validación. Las curvas muestran que la validación deja de mejorar entre las épocas 15 y 59, y ahí se corta.

**9. ¿Qué limitaciones tiene el dataset?**
- Un solo año (52 semanas) y solo 3 categorías originales.
- Unas 3 transacciones por semana y serie, así que las series son muy ruidosas.
- Algunas semanas en 0 (tratadas como cero ventas).
- No trae promociones, festivos ni precios de la competencia.
- No permite medir estacionalidad anual.

**10. ¿Por qué no reentrenaron con train+val para pronosticar la próxima semana?**
Lo probamos con un backtest sin tocar prueba: reentrenar bajaba el MAE solo 0.03 y subía el RMSE 0.12. Como no hay beneficio claro, preferimos que el modelo que pronostica sea exactamente el que evaluamos.

**11. ¿Por qué no un encoder-decoder?**
El enunciado pide predecir una sola semana. El encoder-decoder sirve para generar secuencias de varias semanas; aquí añadiría parámetros sin beneficio, con muy pocos datos.
