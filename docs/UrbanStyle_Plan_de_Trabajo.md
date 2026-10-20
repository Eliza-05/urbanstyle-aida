# UrbanStyle — Plan de trabajo del proyecto final (AIDA_M)

> Documento **base** de acuerdos del equipo, para hablar el mismo idioma antes de programar.
> Dice **qué** hay que hacer, **quién** lo hace y **qué cuidar**. El **cómo** lo investiga y decide cada responsable en su tarea, revisando el material de clase y las fuentes que necesite.
> Todo requisito sale del enunciado `UrbanStyle_Project_Plan-2026_2.pdf`.

**Cómo leer este documento:**

| Etiqueta | Significado |
|---|---|
| **Obligatorio** | Lo exige el enunciado o la rúbrica |
| **Bonus** | Aparece como opcional en el enunciado |
| **[Decisión del equipo]** | Elección nuestra; **no** es exigencia del profesor |
| **⚠️ Pendiente** | Falta definirlo (el equipo o el profesor) |

- **Equipo:** 2 personas
- **Entrega:** ~7 de noviembre de 2026 (fecha provisional: el enunciado dice que la fecha se acuerda en clase).
- **Meta:** 100 % de la rúbrica y todas las bonificaciones, con un proyecto que ambos sepamos sustentar.
- **Principio:** primero asegurar lo **obligatorio**, bien hecho y bien evaluado; después los bonus. Una tarea no está terminada solo porque "corre sin errores".

---

## 1. Contexto del proyecto (según el enunciado)

**UrbanStyle** es un retailer de moda; las cuatro tareas giran en torno a este negocio.
- **Pygame es obligatorio** para la interactividad (botones y sliders).
- **El foco central son los algoritmos:** búsqueda, pronóstico, RL y GAN.

**Dataset recomendado:** [Retail Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)
**Proyecto de referencia sugerido:** [KG0116/store-sim](https://github.com/KG0116/store-sim) (simulación de flujo de clientes en tiendas, como inspiración).

### 1.1 Lo que contiene el dataset

Columnas: `Transaction ID`, `Date` (2023), `Customer ID`, `Gender`, `Age`, `Product Category`, `Quantity`, `Price per Unit`, `Total Amount`.

⚠️ **Son 9 columnas pero solo 3 categorías de producto** (Beauty, Clothing, Electronics). La Tarea B pide 5–8 categorías (ver §5.2).

---

## 2. Rúbrica oficial (copiada del enunciado)

| Tarea | Peso | Criterios obligatorios | Opcional (bonus) |
|---|---|---|---|
| **A — Pathfinding** | 15 % | Modelo de grid, BFS y A\* implementados, rutas visualizadas, zonas de alto tráfico identificadas | Mapa de calor superpuesto, experimentos con distribuciones de tienda |
| **B — Forecasting** | 20 % | Preparación de series de tiempo, modelo LSTM funcionando, MAE y RMSE reportados | Comparación con línea base, exportar CSV |
| **C — RL Pricing** | 25 % | Entorno Gym construido, agente RL entrenado (stable-baselines3), ingresos vs. precio fijo mostrados | Multiproducto o varios algoritmos de RL |
| **D — GANs** | 20 % | GAN pequeña entrenada, imágenes de muestra generadas | GAN condicional (cGAN) o póster promocional |
| **Integración Pygame** | 10 % | Botones/sliders para interactividad, al menos una visualización por tarea | Pulido extra de la interfaz |
| **Reporte y código** | 10 % | Reporte claro, código reproducible, presentación corta | — |

**[Decisión del equipo]** Haremos todos los opcionales, incluidas las dos alternativas donde el enunciado dice "o" (C: multiproducto **y** varios algoritmos; D: cGAN **y** póster). El enunciado no dice si hacer las dos da más puntos, así que se hacen **solo después** de tener lo obligatorio funcionando.

---

## 3. Tecnologías y por qué

| Componente | Tecnología | Usada en |
|---|---|---|
| Lenguaje | Python | Todo |
| Interfaz e interactividad | **Pygame** (obligatorio) | A, B, C, D y menú |
| Redes neuronales | **PyTorch** | B (LSTM), D (GAN / cGAN) |
| Aprendizaje por refuerzo | **Gymnasium + stable-baselines3** (obligatorio SB3) | C |
| Datos | Pandas + NumPy | B, C (A opcional) |
| Escalado y métricas | scikit-learn | B |
| Gráficas | Matplotlib (mostradas dentro de Pygame) | B, C y reporte |
| Imágenes para la GAN | torchvision (Fashion-MNIST) | D |
| Pruebas | pytest | Todas |

**¿Por qué PyTorch y no Keras para la LSTM?** [Decisión del equipo]
- El enunciado **no exige** framework para B ni D; para C exige stable-baselines3, que **funciona sobre PyTorch**.
- Con Keras tendríamos **dos frameworks** pesados que instalar y explicar.
- PyTorch **se vio en clase** (`Intro_to_pytorch.ipynb`, semana 6; taller DCGAN, semana 10). Los notebooks de LSTM en Keras sirven como guía de la lógica; solo cambia la sintaxis.

---

## 4. Reparto de responsabilidades

| Elemento | Responsable |
|---|---|
| Tarea A completa, con su pantalla Pygame | **Sebastián** |
| Tarea C completa, con su pantalla Pygame | **Sebastián** |
| Tarea B completa, con su pantalla Pygame | **Elizabeth** |
| Tarea D completa, con su pantalla Pygame | **Elizabeth** |
| Menú principal de integración (`main.py`) | **Elizabeth** |
| Módulo común de datos y de interfaz (`common/data.py`, `common/ui.py`) | **Elizabeth**, con revisión de Sebastián |
| Configuración común (`common/config.py`) | Ambos |
| Consultar al profesor lo de las categorías de B y ajustar los datos según su respuesta | **Elizabeth** |
| Pruebas de cada tarea | Su responsable |
| Integración final, pruebas finales, reporte y presentación | Ambos |

**Balance:** cada persona tiene **40 %** de la nota en tareas, y cada bloque tiene su propia dificultad y riesgo.

| | Sebastián (A + C) | Elizabeth (B + D + menú) |
|---|---|---|
| Dificultad principal | Diseñar el entorno de RL y cómo reacciona la demanda al precio (C) | Construir una GAN y una cGAN, tema que el equipo no practicó (D) |
| Riesgo principal | Un entorno mal diseñado hace que el agente aprenda algo absurdo | **Los datos de B:** solo hay 3 categorías; según el profesor, puede tocar combinar series, cambiar de dataset o rehacer la preparación. Además hay pocas semanas para la LSTM |
| Carga de Pygame | Alta: clientes animados, mapa de calor, ingresos en vivo | Media: pantallas de B y D, más el **menú que integra las 4 tareas** |
| Trabajo transversal | — | Módulo común y gestión con el profesor |

Ninguno de los dos bloques es "el fácil".

- Ser responsable de `common/` **no** significa hacer la lógica de las tareas del otro: el módulo de datos solo tiene funciones generales, y cada persona construye sus propias pantallas con los componentes comunes.
- ⚠️ **Ambos deben poder explicar las 4 tareas.** El profesor puede preguntarle a cualquiera por cualquier tarea; hay que hacer una sesión cruzada antes de la entrega.

---

## 5. Detalle por tarea

Cada tarea indica: lo que **pide el enunciado**, lo **importante a cuidar** y el **material de clase** que conviene revisar. El material se descarga desde **Teams**, en la semana indicada. Si algo no se vio en clase, se pueden usar fuentes externas **siempre que se pueda sustentar**.

---

### 5.1 Tarea A — Customer Pathfinding & Store Layout (Sebastián) · 15 %

**Objetivo (enunciado):** mostrar cómo se mueven los clientes dentro de la tienda e identificar puntos congestionados.

**Obligatorio (enunciado):**
1. Representar la tienda como grid (p. ej. 8×6).
2. Mapear categorías de producto a secciones.
3. Los clientes entran y se mueven hacia las secciones.
4. Implementar al menos **BFS y A\***.
5. Visualizar las zonas de alto tráfico.

**Pygame (enunciado):** grid y clientes en movimiento, con sliders/botones para cambiar el **algoritmo** o el **número de clientes**.
**Bonus (enunciado):** mapa de calor superpuesto; probar distintas distribuciones de tienda.

**Importante a cuidar:**
- Que todas las secciones sean alcanzables en cada layout, y que un destino sin ruta no rompa la simulación.
- Separar la lógica (grid y búsqueda) de la animación en Pygame.
- Que la velocidad de los clientes no dependa de los FPS.
- Medir el tráfico de forma objetiva (p. ej. visitas por celda), no "a ojo".
- Comparar BFS y A\* en **las mismas condiciones** (mismo grid, origen y destino). Con movimientos de costo uniforme y una heurística admisible, ambos deben encontrar rutas del **mismo largo**. Se miden y **reportan** longitud, nodos expandidos y tiempo, sin dar por hecho de antemano que A\* siempre expande menos nodos.
- ⚠️ Pendiente: decidir si varios clientes pueden ocupar la misma celda. Lo más simple es permitirlo y medir la congestión; bloquear celdas obliga a re-planear rutas en tiempo real.

**Material de clase:**

| Tema | Dónde |
|---|---|
| Búsqueda: BFS, DFS, UCS, Dijkstra, complejidad | Semana 2 (Teams): `AIDA week 02.pdf` |
| Tarea de clase: grid leído desde una matriz, con obstáculos | Semana 2 (Teams): `AIDA week 02.pdf` (sección Homework) |
| **A\*** con cola de prioridad, f(n)=g(n)+h(n), heurística Manhattan | Semana 9 (Teams): `Artificial_Intelligence_Sprite_Simulation_final.pdf`, **Milestone 4** |
| Agentes que siguen rutas; varios agentes | mismo PDF, **Milestones 5 y 6** |
| Botones y sliders en Pygame | mismo PDF, **Milestone 7** |
| Simulación animada con controles y exportación de resultados | Semana 2 (Teams): `konigsberg_bridge_pygame.zip` |
| Fundamentos de Pygame | Semana 1 (Teams): `pygame.pdf` y los .zip de *Project files* |

---

### 5.2 Tarea B — Demand Forecasting con LSTM (Elizabeth) · 20 %

**Objetivo (enunciado):** predecir ventas futuras para apoyar decisiones de inventario.

**Obligatorio (enunciado):**
1. Seleccionar 5–8 categorías.
2. Preprocesar como series de tiempo **semanales**.
3. Entrenar una LSTM pequeña o un encoder-decoder.
4. Predecir las ventas de la **siguiente semana**.
5. Evaluar con **MAE y RMSE**.

**Pygame (enunciado):** gráfica del pronóstico, con sliders para elegir la **categoría** o la **duración del entrenamiento**.
**Bonus (enunciado):** línea base de media móvil; exportar predicciones a CSV.

**Importante a cuidar:**
- ⚠️ **Pendiente crítico — Elizabeth lo consulta con el profesor.** El dataset solo tiene 3 categorías. Propuesta: formar 5–8 **series** cruzando columnas reales (p. ej. Categoría × Género = 6). En el reporte se llaman series o segmentos, no categorías nuevas. Si el profesor no lo acepta, se busca otro dataset y se documenta el cambio.
- ⚠️ Pendiente: decidir si se pronostican **unidades** o **ingresos** (no es lo mismo).
- **Semanas sin registros:** primero verificar la cobertura de fechas del dataset. Tratar esas semanas como cero ventas es una **hipótesis de trabajo** que debe justificarse en el reporte: que no aparezca una semana podría significar cero ventas o falta de información.
- Hay pocas semanas por serie: usar una red pequeña y vigilar el sobreajuste.
- Evitar la fuga de información: dividir los datos **por orden cronológico** y ajustar el escalado solo con los datos de entrenamiento.
- MAE y RMSE se reportan en unidades reales (revirtiendo el escalado).
- LSTM y media móvil se comparan sobre **las mismas semanas** de prueba. Si la media móvil gana, se reporta así.
- El enunciado pide categoría **o** duración del entrenamiento, no ambos. El control principal es la categoría. Si también se agrega el de duración, debe cargar **modelos distintos realmente entrenados** con esas duraciones, guardados en archivos separados.

**Material de clase:**

| Tema | Dónde |
|---|---|
| RNN, **LSTM**, GRU, desvanecimiento del gradiente, encoder-decoder | Semana 6 (Teams): `AIDA_RNN.pdf` |
| Pérdida, sobreajuste, early stopping, dropout | Semana 6 (Teams): `ANN.pdf` y `AIDA_RNN.pdf` |
| **Series de tiempo con LSTM**: escalado, ventanas, división train/test, RMSE (en Keras; misma lógica en PyTorch) | Semana 6 (Teams): `time-series-analysis-using-lstm-keras.rar` y `Stock market task RNN LSTM.rar` |
| PyTorch: tensores, `nn.Module`, optimizadores, ciclo de entrenamiento | Semana 6 (Teams): `Intro_to_pytorch.ipynb` |
| Encoder-decoder con LSTM | Semana 6 (Teams): `LLM_Keras_RNN_LSTM_translation.ipynb` |

---

### 5.3 Tarea C — Dynamic Pricing con RL (Sebastián) · 25 %

**Objetivo (enunciado):** ajustar precios dinámicamente para maximizar ingresos.

**Obligatorio (enunciado):**
1. Definir un entorno Gym:
   - **estado:** precio, segmento, ventas recientes;
   - **acciones:** ajustar el precio;
   - **recompensa:** ingreso.
2. Entrenar un agente con **stable-baselines3 (DQN o PPO)**.
3. Ejecutar episodios y medir el ingreso.
4. Comparar contra **precio fijo**.

**Pygame (enunciado):** gráfica de ingresos **en vivo**, sliders para fijar el **precio inicial** o **reiniciar** el entorno.
**Bonus (enunciado):** varios productos; varios algoritmos de RL.

**Importante a cuidar:**
- **El punto central de la tarea es cómo reaccionan los clientes al precio.** Si la demanda no cambia con el precio, el agente aprende a subirlo al máximo y no resuelve nada. Sebastián define una regla de demanda coherente (más precio → menos compras), y los supuestos quedan documentados.
  > 💡 **Sugerencia (no obligatoria): "disposición a pagar".** Cada cliente tiene un precio máximo que está dispuesto a pagar, y ese máximo varía según su segmento (edad o género). El cliente compra solo si el precio es menor o igual a ese máximo. Así, el ingreso queda como precio × número de clientes que aceptan ese precio. Un precio muy alto espanta a casi todos y uno muy bajo deja dinero sobre la mesa, de modo que el agente tiene que buscar un equilibrio. Es solo una forma posible de construir la regla de demanda: Sebastián puede usar otra si la justifica.
- Para que los precios dinámicos tengan sentido, conviene que distintos segmentos o escenarios tengan distintos precios óptimos. Así un precio fijo no puede ser siempre lo mejor.
- **El dataset solo registra compras**, no a quienes entraron y no compraron. Sirve para fijar escalas (precios reales, cantidades, tipos de comprador), pero **no** revela cómo reaccionarían los clientes a otros precios: esa parte es supuesto de simulación y se declara así.
- Precios siempre dentro de un mínimo y un máximo; recompensa = ingreso (precio × unidades), no utilidad.
- Acciones **discretas** (p. ej. bajar, mantener o subir el precio), para que funcionen tanto DQN como PPO.
- Validar el entorno **antes** de entrenar. La herramienta `check_env` de SB3 verifica que sea compatible, pero no que el agente aprenda bien.
- Comparación justa: mismos escenarios y semillas, suficientes episodios, promedio y variabilidad, y el **mejor** precio fijo como referencia (no uno arbitrario). No afirmar que RL siempre gana.
- La pantalla carga agentes ya entrenados; nunca entrena dentro de Pygame.

**Material de clase:**

| Tema | Dónde |
|---|---|
| MDP, Bellman, Monte Carlo, TD, **Q-learning**, Gymnasium, Deep Q-Learning | Semana 3 (Teams): `Reinforcement Learning.pdf` |
| ⭐ **El ejemplo más parecido**: agente que aprende a **elegir precios** en un mercado (precios discretos, recompensa = ganancia, replay buffer, ε-greedy) | Semana 4 (Teams): `RL_Market_Example.pdf` |
| **Entorno Gymnasium propio** (`reset`, `step`, espacios, recompensas) | Semana 9 (Teams): `Artificial_Intelligence_Sprite_Simulation_final.pdf`, **Milestone 8** |
| **PPO** (para entender qué hace SB3 por dentro) | Semana 9 (Teams): `Artificial_Intelligence_MARL_Pygame_final.pdf`, **Milestone 5** |
| Botones y sliders en Pygame | Sprite Simulation PDF, **Milestone 7** |
| Libro de referencia (Sutton & Barto) | Semana 3 (Teams): `RLbook2018.pdf` |

**Externo necesario:** stable-baselines3 **no se vio en clase** pero lo exige el enunciado. Hay que estudiar su documentación oficial y saber explicar qué hace por dentro con lo visto en clase.

---

### 5.4 Tarea D — GAN para imágenes promocionales (Elizabeth) · 20 %

**Objetivo (enunciado):** crear nuevas imágenes promocionales de moda, con [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

**Obligatorio (enunciado):**
1. Entrenar una GAN muy pequeña (p. ej. **DCGAN**).
2. **Generador:** ruido → imagen 28×28.
3. **Discriminador:** clasifica real/falso.
4. Entrenar unas pocas épocas y guardar las salidas.

**Pygame (enunciado):** botón para **generar imágenes** y slider de **semilla del ruido**.
**Bonus (enunciado):** GAN condicional (cGAN), o combinar imágenes en un **póster promocional**.

**Importante a cuidar:**
- El entrenamiento de una GAN es inestable. Hay que vigilar si el discriminador domina o si el generador produce imágenes casi todas iguales.
- La calidad se juzga **viendo las imágenes**, no solo las pérdidas. Guardar muestras durante el entrenamiento con un ruido fijo, para ver el progreso.
- En la cGAN, comprobar que la clase elegida (camiseta, pantalón, zapatilla…) **realmente** cambia lo que se genera. Su ventaja es controlar la clase, no necesariamente más realismo.
- Fashion-MNIST es 28×28 en escala de grises: no se esperan imágenes fotorrealistas. El póster se presenta como composición de imágenes generadas, distinguiéndolas de las reales.
- En Pygame: selector GAN/cGAN, selector de prenda para la cGAN, y generación con el modelo ya entrenado (nunca entrenar al presionar el botón).

**Material de clase** (el equipo no hizo el taller de la semana 10, así que esta es la lectura prioritaria):

| Tema | Dónde |
|---|---|
| ⭐ **Guía paso a paso de DCGAN con Fashion-MNIST en PyTorch** | Semana 10 (Teams): `W2_DCGAN.pdf` |
| Utilidades del taller (carga de Fashion-MNIST, guardar grillas, semillas) | Semana 10 (Teams): `UPLOAD_TO_STUDENTS.zip` |
| Teoría: entropía cruzada binaria, minimax de la GAN | Semana 7 (Teams): `Generative AI.pdf` |
| Tutorial oficial DCGAN | Semana 7 (Teams): `dcgan_faces_tutorial.ipynb` |
| Espacio latente (autoencoder) | Semana 10 (Teams): `W1_Autoencoder.pdf` |
| PyTorch con Fashion-MNIST | Semana 6 (Teams): `Intro_to_pytorch.ipynb` |

**Externo útil (la cGAN no se vio en clase):** Mirza & Osindero (2014), *Conditional Generative Adversarial Nets*.

---

### 5.5 Integración Pygame — menú principal (Elizabeth) · 10 %

**Enunciado:** botones y sliders para la interactividad, **al menos una visualización por tarea**. Bonus: pulido extra de la interfaz.

**Acuerdos [Decisión del equipo]:**
- `main.py` abre un menú con acceso a las 4 tareas; cada pantalla puede volver al menú.
- Todas las pantallas se integran de la **misma forma**. ⚠️ Pendiente: ambos acuerdan esa forma antes de programar las pantallas.
  > 💡 **Sugerencia:** que cada `screen.py` tenga una clase con tres métodos: `handle_event` (clics, teclas), `update` (avanza la lógica) y `draw` (dibuja). Más un botón "Volver" al menú. Así el menú abre cualquier pantalla sin conocer su contenido.
  > En el taller Sprite City (semana 9, Milestone 7), el profesor separa así botones y sliders, en clases `Button` y `Slider` con métodos `draw`, `clicked` y `handle_event`. Sirve de referencia para `common/ui.py`.
- Pygame **solo carga** modelos ya entrenados; nunca entrena.
- **Ningún control decorativo:** cada botón o slider cambia algo real.
- Mismo estilo visual en las 4 pantallas (colores, fuentes, botones), para que UrbanStyle se vea como **una sola aplicación**.

**Material de clase:** menú en Pygame (Semana 1, `menu.zip`); botones y sliders (Semana 9, Sprite Simulation, Milestone 7); controles y escalado de ventana (Semana 2, `konigsberg_bridge_pygame.zip`).

---

### 5.6 Reporte y código · 10 % (ambos)

**Enunciado:** reporte claro, código reproducible, presentación corta.

- Cada persona escribe la sección de sus tareas; Elizabeth arma la estructura del documento.
- Por cada tarea: problema, decisiones, algoritmo, resultados (métricas y gráficas), comparación y limitaciones.
- Se declaran los supuestos (series de B, demanda simulada de C).

---

## 6. Datos compartidos y dependencias entre tareas

**Las cuatro tareas son independientes:** ninguna necesita ejecutar el algoritmo de otra, y no hay que hacerlas en orden.

| Tarea | Fuente de datos |
|---|---|
| A | Simulación propia de la tienda (opcionalmente, proporciones de compra del dataset) |
| B | Dataset de ventas (series semanales) |
| C | Demanda **simulada**, con escalas tomadas del dataset de ventas. **No** usa la LSTM de B |
| D | Fashion-MNIST, independiente del dataset de ventas |

**Lo único compartido es la lectura y limpieza del dataset de ventas**, en `common/data.py`, para que B y C trabajen con los mismos datos limpios. No se crean dependencias artificiales: la integración real es la aplicación UrbanStyle y su caso de negocio.

> 💡 **Ejemplo orientativo** de lo que podría ofrecer `common/data.py` (los nombres y funciones exactas los define Elizabeth):
> ```python
> load_sales()        # lee el CSV, convierte fechas, valida columnas
> add_segments(df)    # crea rangos de edad y series tipo "Clothing-Female"
> weekly_series(df)   # ventas por semana y por serie      → la usaría B
> pricing_stats(df)   # precios y compradores por segmento → la usaría C
> ```
> El modelo de demanda de C **no** va aquí: vive en la carpeta de la Tarea C.

---

## 7. Estructura del repositorio

```
urbanstyle/
├── README.md                 ← instalación, cómo entrenar y cómo ejecutar
├── requirements.txt
├── .gitignore
├── main.py                   ← menú principal (Elizabeth)
├── data/
│   ├── raw/                  ← CSV original, nunca se modifica
│   └── processed/            ← datos transformados
├── common/                   ← código compartido
│   ├── data.py               ← carga y limpieza del dataset (Elizabeth)
│   ├── ui.py                 ← botones, sliders, colores, fuentes (Elizabeth)
│   └── config.py             ← rutas, semillas, constantes (ambos)
│
├── task_a_pathfinding/       ← Sebastián
│   ├── store.py              ← grid, secciones y layouts
│   ├── search.py             ← BFS y A*
│   ├── simulation.py         ← clientes en movimiento, conteo de tráfico
│   └── screen.py             ← pantalla Pygame
│
├── task_b_forecasting/       ← Elizabeth
│   ├── model.py              ← la LSTM
│   ├── train.py              ← entrena y guarda (se ejecuta una vez)
│   ├── evaluate.py           ← MAE, RMSE, media móvil, exportar CSV
│   └── screen.py
│
├── task_c_pricing/           ← Sebastián
│   ├── demand.py             ← regla de demanda (cómo reaccionan al precio)
│   ├── env.py                ← entorno Gymnasium (uno y varios productos)
│   ├── train.py              ← entrena DQN y PPO con SB3
│   ├── evaluate.py           ← RL vs. mejor precio fijo, DQN vs. PPO
│   └── screen.py
│
├── task_d_gan/               ← Elizabeth
│   ├── models.py             ← generador y discriminador (GAN y cGAN)
│   ├── train.py
│   ├── poster.py             ← póster promocional
│   └── screen.py
│
├── models/                   ← modelos FINALES entrenados (sí se suben)
│   ├── lstm_forecast.pt
│   ├── dqn_pricing.zip       ← SB3 guarda en .zip
│   ├── ppo_pricing.zip
│   ├── pricing_config.json   ← parámetros de la demanda simulada
│   ├── gan_generator.pt
│   └── cgan_generator.pt
│
├── outputs/                  ← gráficas, CSV, imágenes y póster
│   └── task_a/  task_b/  task_c/  task_d/
├── tests/                    ← pruebas rápidas con pytest
│   └── test_search.py  test_data.py  test_env.py  test_gan.py
└── report/                   ← informe y presentación
```

> 💡 **Los archivos internos son una sugerencia.** La idea de fondo sí es importante: separar en cada tarea el algoritmo/modelo, el entrenamiento, la evaluación y la pantalla. Cada responsable puede renombrar o reorganizar sus archivos si lo justifica.

**Reglas de la estructura:**
- **Entrenar y usar son momentos separados.** Los modelos se entrenan una vez con su propio script y se guardan en `models/`. Al ejecutar `python main.py`, las pantallas solo los cargan. Así el profesor puede clonar, instalar y abrir la app **sin entrenar nada**.
- **Junto a cada modelo se guarda todo lo necesario para usarlo igual que en el entrenamiento:** escalador de la LSTM, configuración, etiquetas de la cGAN, parámetros del entorno de C, etc.

  > 💡 **Así aparece en el ejemplo del profesor** (taller W2_DCGAN, semana 10): el checkpoint guarda los pesos **y** la configuración en un solo archivo. Es una referencia, no la única forma:
  > ```python
  > torch.save({"G": generator.state_dict(), "D": discriminator.state_dict(),
  >             "nz": cfg["nz"], "base": cfg["base"]}, ckpt / "dcgan.pt")
  > ```
  > La misma idea sirve para la LSTM (pesos + parámetros del escalador + tamaño de ventana + nombres de las series) y para la cGAN (pesos + `nz` + lista de clases). Para el escalador también se puede guardar el objeto completo con `joblib`.
- **Qué se sube a Git:**
  - Sí: código, CSV de ventas (pesa pocos KB), modelos finales, gráficas del reporte.
  - No: entornos virtuales, cachés, la carpeta de Fashion-MNIST (se descarga sola), checkpoints intermedios y logs.
  - ⚠️ Pendiente: verificar el tamaño de los modelos finales antes de subirlos.
- **Reproducibilidad:**
  - semillas fijas;
  - rutas relativas (nunca rutas de nuestros computadores);
  - que funcione **sin GPU**;
  - ⚠️ Pendiente: fijar la versión de Python y de las dependencias (sugerencia: Python 3.13, la versión con la que se verificó el proyecto).
  - Antes de entregar, probar todo desde un clon limpio.

---

## 8. Cómo validamos que algo está bien

Que el código corra no significa que esté bien. Cada tarea se revisa en cuatro niveles:

| Nivel | Pregunta | Ejemplo |
|---|---|---|
| 1. Funcional | ¿Corre y entrega la salida esperada? | El generador produce imágenes 28×28 |
| 2. Algorítmica | ¿El algoritmo cumple sus propiedades? | BFS y A\* dan rutas del mismo largo con costo uniforme |
| 3. Experimental | ¿Las métricas y comparaciones son justas? | LSTM vs. media móvil sobre las mismas semanas |
| 4. Visual | ¿La interfaz muestra bien los resultados? | El mapa de calor coincide con los conteos |

- Las pruebas automáticas (`pytest`) cubren los niveles 1 y 2, son rápidas y **no entrenan modelos**.
- Los niveles 3 y 4 se verifican con los scripts de evaluación y revisando la app.
- Las evidencias (gráficas, tablas, grillas de imágenes) se guardan en `outputs/` para el reporte.

---

## 9. Reglas de trabajo en equipo

1. Cada persona trabaja **solo en sus carpetas**.
2. Los cambios en `common/` se hacen en una rama propia y el otro los revisa antes de unirlos (*pull request*).
3. Primero **validar los fundamentos** de cada tarea (el grid de A, las series de B, el entorno de C, la arquitectura mínima de D), después entrenar, después comparar y por último pulir.
4. Todo lo que no se vio en clase debe poder **explicarse línea por línea**.
5. Ningún comportamiento se presenta como "aprendido del dataset" si fue un supuesto del equipo.

---

## 10. Hitos

Cada persona organiza su tiempo como prefiera, con tres condiciones:

| Hito | Cuándo | Qué debe estar |
|---|---|---|
| **1. Versión obligatoria funcionando** | **Final de octubre** | Las 2 tareas de cada persona cumplen todo lo obligatorio de la rúbrica, evaluadas y con su pantalla. **A partir de ahí, los bonus que se tengan hasta ese punto.** |
| **2. Pruebas e integración** | **1–7 de noviembre** | Todo integrado en el menú, pruebas, ejecución desde un clon limpio, reporte, presentación y sesión cruzada |
| Revisión temprana | Lo antes posible | Revisar juntos el entorno de C y los datos de B antes de invertir muchas horas en entrenar |

---

## 11. Checklist de finalización

Una tarea está terminada **solo** si se marcan todas sus casillas.

**Tarea A**
- [ ] Grid y layouts válidos (todas las secciones alcanzables)
- [ ] BFS y A\* implementados; manejan destinos sin ruta
- [ ] Clientes que entran y se mueven a las secciones
- [ ] Zonas de alto tráfico identificadas con una medida objetiva
- [ ] Comparación BFS vs. A\* (longitud, nodos expandidos, tiempo) guardada
- [ ] Bonus: mapa de calor y varios layouts
- [ ] Pantalla con cambio de algoritmo y de número de clientes
- [ ] Pruebas pasan; evidencias guardadas

**Tarea B**
- [ ] Series y dataset acordados con el profesor; hipótesis sobre semanas vacías justificada
- [ ] 5–8 series semanales; división cronológica sin fuga de información
- [ ] LSTM entrenada y guardada con lo necesario para reutilizarla
- [ ] Pronóstico de la semana siguiente; MAE y RMSE en unidades reales
- [ ] Bonus: media móvil sobre las mismas semanas y exportación CSV
- [ ] Pantalla con selector de categoría (y control de duración, si se implementa)
- [ ] Pruebas pasan; evidencias guardadas

**Tarea C**
- [ ] Supuestos de demanda documentados; el precio afecta realmente las ventas
- [ ] Entorno con estado, acciones y recompensa del enunciado; `check_env` pasa
- [ ] Agente(s) SB3 entrenados y guardados, con curvas de aprendizaje
- [ ] Comparación justa contra el mejor precio fijo
- [ ] Bonus: varios algoritmos y multiproducto
- [ ] Pantalla con ingresos en vivo, precio inicial y reset
- [ ] Pruebas pasan; evidencias guardadas

**Tarea D**
- [ ] GAN entrenada; imágenes de muestra y evolución guardadas
- [ ] Generador guardado con su configuración
- [ ] Bonus: cGAN que respeta la clase pedida, y póster promocional
- [ ] Pantalla con botón de generar, semilla, selector GAN/cGAN y de prenda
- [ ] Pruebas pasan; evidencias guardadas

**Integración y entrega (ambos)**
- [ ] El menú abre las 4 pantallas y todas vuelven sin errores
- [ ] Estilo visual unificado
- [ ] `python main.py` funciona en un clon limpio, sin GPU y sin entrenar
- [ ] README, reporte y presentación terminados
- [ ] Sesión cruzada hecha: cada uno puede explicar las 4 tareas

---

## 12. Pendientes antes de empezar

| Pendiente | Prioridad | Responsable |
|---|---|---|
| Preguntar al profesor si acepta series combinadas (p. ej. Categoría × Género) para las 5–8 categorías de B | Crítica | Elizabeth |
| Confirmar con el profesor la fecha de entrega y si cuentan las dos alternativas de bonus donde dice "o" | Alta | Elizabeth |
| Revisar cobertura de fechas y número de semanas útiles por serie; decidir unidades vs. ingresos | Alta | Elizabeth |
| Definir la regla de demanda de C y sus supuestos | Crítica | Sebastián |
| Decidir si en A varios clientes pueden ocupar la misma celda | Media | Sebastián |
| Acordar cómo se integran las pantallas al menú | Alta | Ambos |
| Fijar versión de Python y dependencias | Media | Ambos |
| Crear el repositorio en GitHub con la estructura del §7 | Alta | Ambos |
