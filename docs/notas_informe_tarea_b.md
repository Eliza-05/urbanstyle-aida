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
- **¿Tiene sentido un LSTM?** El enunciado lo exige y se entrena, pero con este ruido el techo de cualquier modelo es bajo. **Riesgos:** sobreajuste (pocas ventanas por serie) y que el modelo termine aprendiendo solo la media. Mitigaciones: un modelo compartido entre las 6 series (6× más ejemplos), una red pequeña, dropout, early stopping y una comparación honesta con las líneas base en las mismas semanas.

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

`python -m task_b_forecasting.tune` (152 s en CPU) compara, **solo con validación** y con 3 semillas por configuración:
- **(A) por serie:** 6 LSTM independientes. Entrada: la serie sola.
- **(B) compartida:** una LSTM para las 6 series, con el one-hot de la serie concatenado a cada paso. Se eligió one-hot y no un *embedding* porque es lo más simple y se vio en clase.

Grilla: ventana {4, 8} × ocultas {8, 16, 32} × dropout {0, 0.2} × lr {0.001, 0.005} = 24 configuraciones por opción (`outputs/task_b/tuning_results.csv`).

| Opción | Mejor configuración | MAE val (media ± desv., 3 semillas) | RMSE val | Mejor época media |
|---|---|---|---|---|
| **A por serie** | w=8, h=16, d=0, lr=0.005 | **3.825 ± 0.064** | 4.508 | 15 |
| B compartida | w=8, h=16, d=0, lr=0.005 | 3.968 ± 0.158 | 4.807 | 2 |

- Las **12 mejores** configuraciones de toda la grilla son de la opción A.
- La compartida casi no aprende: su mejor época de validación es la 0–6 con lr=0.005, y sigue en ~4.0 de MAE con lr=0.001. Una explicación probable es que la autocorrelación es distinta en cada serie (EDA §3), y forzar una sola dinámica mezcla patrones que se contradicen.
- **Consecuencia:** el checkpoint guarda 6 conjuntos de pesos en un solo archivo.

### 6.3 Decisión: configuración final w=4, h=8, dropout=0.2, lr=0.005

Muchas configuraciones de A empatan dentro del ruido entre semillas (3.825 a 3.858). Se aplica la **regla de parsimonia**: entre las que quedan a menos de una desviación estándar de la mejor (3.825 + 0.064 = 3.889), se elige la red más pequeña. Ante un empate, menos parámetros significa menos riesgo de sobreajuste.

| Configuración (A) | MAE val | Parámetros por LSTM |
|---|---|---|
| w=8, h=16, d=0 (mejor) | 3.825 ± 0.064 | 1 233 |
| **w=4, h=8, d=0.2 (elegida)** | **3.833 ± 0.036** | **361** |

- **Ventana de 4 semanas (~1 mes):** la de 8 no mejora fuera del ruido, y con 4 cada serie tiene 32 ventanas de entrenamiento en vez de 28.
- **8 unidades ocultas:** red pequeña para ~32 ejemplos por serie.
- **dropout 0.2:** regulariza, y en la grilla no empeora el resultado.

### 6.4 Entrenamiento final (`python -m task_b_forecasting.train`)

- **Tiempo:** 2.6 s en CPU para las 6 LSTM.
- **Mejor época por serie:** 15, 59, 26, 16, 26 y 24, con early stopping a las 46–90 épocas.
- **Curvas** (`outputs/task_b/loss_curve.png`): la pérdida de entrenamiento baja y se estabiliza en ~0.08. La de validación baja hasta la mejor época y luego queda plana o sube un poco (Clothing-Male sube entre las épocas 15 y 30). Es una señal leve de sobreajuste que el early stopping corta. La validación queda por debajo de entrenamiento porque el dropout solo actúa al entrenar y porque la pérdida de entrenamiento es un promedio durante la época.
- **Checkpoint** `models/lstm_forecast.pt` (21 KB): pesos de las 6 LSTM, `scaler_min`/`scaler_max`, ventana, nombres de las series, hiperparámetros, límites del split, k de la media móvil, semilla y mejores épocas. Al recargarlo, las predicciones son **idénticas** (`np.array_equal` = True, comprobado en `train.py` y en `tests/`).

## 7. Resultados (`python -m task_b_forecasting.evaluate --seeds 5`)

Prueba = las últimas 8 semanas (2023-11-05 a 2023-12-24) × 6 series = **48 predicciones, las mismas para los 3 modelos**. Métricas en **unidades por semana** (con el escalado ya invertido). Las métricas globales se recalcularon con `sklearn.metrics` y coinciden (`verify_with_sklearn` = True).

### 7.1 Checkpoint final (semilla 42)

| Serie | LSTM MAE | LSTM RMSE | Media móvil (k=8) MAE | MM RMSE | Ingenua MAE | Ingenua RMSE |
|---|---|---|---|---|---|---|
| Clothing-Male | 3.485 | 4.004 | **2.969** | **4.002** | 3.500 | 4.444 |
| Clothing-Female | **3.950** | **4.228** | 4.062 | 4.368 | 5.875 | 6.452 |
| Beauty-Male | **2.453** | **3.320** | 2.594 | 3.815 | 4.000 | 4.743 |
| Beauty-Female | 2.930 | 3.536 | **2.312** | **2.651** | 3.500 | 3.841 |
| Electronics-Male | 3.697 | **4.319** | **3.688** | 4.606 | 5.375 | 6.031 |
| Electronics-Female | **3.892** | **5.080** | 4.422 | 5.837 | 6.500 | 7.194 |
| **GLOBAL** | 3.401 | **4.121** | **3.341** | 4.320 | 4.792 | 5.579 |

### 7.2 Variabilidad entre 5 semillas (42–46), métricas globales en prueba

| Modelo | MAE (media ± desv.) | RMSE (media ± desv.) |
|---|---|---|
| LSTM | 3.430 ± 0.039 | **4.164 ± 0.051** |
| Media móvil k=8 | **3.341** (determinista) | 4.320 |
| Ingenua | 4.792 | 5.579 |

### 7.3 Interpretación honesta

- **La LSTM y la media móvil quedan prácticamente empatadas.** La media móvil gana en MAE por 0.09 unidades/semana, una diferencia de solo ~2 desviaciones entre semillas. La LSTM gana en RMSE por 0.16, con las 5 semillas por debajo de 4.32: comete **menos errores grandes**. Por serie, cada modelo gana en la mitad de los casos.
- **Las dos le ganan claramente a la ingenua:** ~29 % menos MAE y ~25 % menos RMSE. Copiar la semana anterior es la peor estrategia con series tan ruidosas.
- **La LSTM aprendió el nivel de cada serie, no una dinámica.** Sus predicciones casi no varían de una semana a otra: la desviación estándar de sus pronósticos es de 0.3 a 0.5 unidades, contra 4.3 a 5.2 de las ventas reales y 1.1 a 2.7 de la media móvil. Es coherente con el EDA: si las semanas casi no están correlacionadas, la mejor predicción bajo MSE es un nivel estable, y eso aprendió la red. Este comportamiento suaviza más que la media móvil y explica por qué gana en RMSE.
- En validación, el checkpoint de semilla 42 tiene un MAE de 3.917 contra 3.859 de la media móvil. Su RMSE es menor: 4.564 contra 4.745.
- **Mejoras honestas que se probaron (todas con validación):** ventana 4/8, ocultas 8/16/32, dropout 0/0.2, lr 0.001/0.005, y modelo por serie vs. compartido (§6). Ninguna configuración se separa del resto más allá del ruido entre semillas. Seguir ajustando con 8 semanas de validación sería ajustar ruido, así que se dejó así.

### 7.4 Pronóstico de la próxima semana (semana del 2023-12-31; `outputs/task_b/next_week.csv`)

| Serie | LSTM | Media móvil | Ingenua |
|---|---|---|---|
| Clothing-Male | 5.16 | 7.00 | 1 |
| Clothing-Female | 8.15 | 7.38 | 11 |
| Beauty-Male | 7.68 | 6.00 | 6 |
| Beauty-Female | 6.78 | 8.88 | 13 |
| Electronics-Male | 8.32 | 8.12 | 2 |
| Electronics-Female | 9.96 | 10.00 | 12 |

Para inventario, con un MAE de ~3.4 unidades sobre niveles de 6 a 10, conviene presentar el pronóstico como un **rango** (± el MAE) y no como un número exacto.

### 7.5 Exportación (bonus)

`outputs/task_b/predictions.csv`: 270 filas con columnas `week_start, series, part, actual, lstm, moving_average, naive`. `part` ∈ {train (168), val (48), test (48), next_week (6)}; `actual` está vacío solo en `next_week`. Las filas de train y val se incluyen para graficar; las métricas usan solo `test`. La pantalla tiene un botón que regenera este archivo.

## 8. Pantalla Pygame (`task_b_forecasting/screen.py`)

- **Control del enunciado:** un **slider para elegir la serie** (6 posiciones; también funcionan las flechas ← →). No se agregó el slider de duración de entrenamiento: el enunciado pide categoría **o** duración, y ese control exigiría cargar modelos distintos entrenados con cada duración.
- **Gráfica:** ventas reales de las 52 semanas, predicción de la LSTM (línea azul), media móvil k=8 (línea naranja punteada), zona de prueba sombreada y punto de la próxima semana.
- **Panel derecho:**
  - MAE y RMSE en prueba de LSTM, media móvil e ingenua, para la serie elegida y global.
  - Pronóstico de la próxima semana, LSTM ± su MAE y media móvil.
- **Botones y textos:**
  - Botón **Exportar CSV**, que regenera `outputs/task_b/predictions.csv` y lo confirma en pantalla.
  - Botón **Volver** o tecla ESC.
  - Texto fijo: "3 categorías originales × género = 6 series".
- **Solo carga el modelo:** carga `models/lstm_forecast.pt` y nunca entrena. torch se importa solo al entrar a la pantalla, para que el menú abra rápido, y los resultados quedan en caché. Si falta el checkpoint, muestra el comando para entrenarlo en vez de fallar.
- **Verificación headless** (`SDL_VIDEODRIVER=dummy`): el clic en el slider cambia a Beauty-Female y la flecha derecha a Electronics-Male; la exportación muestra el mensaje y ESC cierra la pantalla. Desde `main.py`, clic en "Tarea B" → slider → ESC → cerrar ventana termina limpio. Las capturas de las 6 series y de la pantalla de error se revisaron a ojo. Captura: `outputs/task_b/screen_task_b.png`.
