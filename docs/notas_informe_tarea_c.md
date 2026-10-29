# Tarea C — Precios dinámicos con aprendizaje por refuerzo: notas para el informe

> Documento de trabajo con supuestos, decisiones (con números) y resultados reales de la Tarea C.
> Números de `python -m task_c_pricing.evaluate` (archivos en `outputs/task_c/metrics/`).

## 1. Requisitos (enunciado `UrbanStyle_Project_Plan-2026_2.pdf`)

**Objetivo:** ajustar precios dinámicamente para maximizar el ingreso.

| # | Obligatorio | Dónde se cumple |
|---|---|---|
| 1 | Entorno Gym: estado (precio, segmento, ventas recientes), acciones (ajustar precio), recompensa (ingreso) | `env.py::PricingEnv` (Gymnasium) |
| 2 | Agente de stable-baselines3 (DQN o PPO) | `train.py`: **DQN y PPO** |
| 3 | Ejecutar episodios y medir el ingreso | `evaluate.py` (200 episodios) |
| 4 | Comparar contra precio fijo | `evaluate.py`: contra el **mejor** precio fijo |
| Pygame | Gráfica de ingresos en vivo; sliders para el precio inicial o reiniciar | `screen.py` |
| Bonus | Varios productos; varios algoritmos de RL | `ppo_pricing_multi.zip` (3 productos); DQN vs. PPO |

Rúbrica: 25 % (la tarea de mayor peso).

## 2. Modelo de demanda (`demand.py`): qué sale del dataset y qué es supuesto

> ⚠️ **El dataset solo registra compras.** Da escalas reales (precios, cantidades, tipos de cliente), pero **no** cómo reaccionaría un cliente a otro precio. La regla de reacción al precio es un **supuesto de simulación** y así se declara.

**Regla: "disposición a pagar".**
- Cada día llegan `N ~ Poisson(40)` clientes interesados en cada producto.
- Cada cliente tiene un precio máximo que está dispuesto a pagar: `WTP ~ LogNormal(mediana = escala[segmento][producto] × precio de referencia, σ = 0,30)`.
- Compra si `precio ≤ WTP`, y lleva de 1 a 4 unidades.
- **Más precio → menos compradores:** `P(compra) = 1 − Φ((ln p − ln mediana)/σ)` es estrictamente decreciente (prueba `test_demand_decreases_with_price`). Por eso el agente no gana subiendo el precio al máximo.

| Parámetro | Valor | Fuente |
|---|---|---|
| Precio de referencia | Ropa 174,29 · Belleza 184,06 · Electrónica 181,90 (precio medio pagado por categoría) | **dataset** (`pricing_stats`) |
| Segmentos | 3 rangos de edad: 18-35, 36-55, 56-64 | **dataset** (columna Age, agrupando los rangos de `common.data`) |
| Participación de cada segmento | 37,4 % · 43,1 % · 19,5 % | **dataset** |
| Unidades por compra (1–4) | distribución real de cada segmento | **dataset** |
| Clientes interesados por día | 40 por producto | supuesto |
| Escala de la disposición a pagar | tabla siguiente | supuesto |
| σ (dispersión) | 0,30 | supuesto |
| Persistencia del segmento | 0,85: mañana predomina el mismo segmento con prob. 0,85; si no, se sortea según la participación real | supuesto |

**Escala de la disposición a pagar (mediana / precio de referencia; supuesto):**

| Segmento | Ropa | Belleza | Electrónica |
|---|---|---|---|
| Jóvenes 18-35 | 0,95 | 1,05 | 1,65 |
| Adultos 36-55 | 1,70 | 1,30 | 1,30 |
| Mayores 56-64 | 1,30 | 1,75 | 0,95 |

**Por qué segmentos con óptimos distintos:**
- En el dataset el precio medio por edad varía poco (de 170,84 a 191,92), así que las diferencias entre segmentos **se construyeron** en el modelo, como pide el plan.
- Sin ellas, un precio fijo sería óptimo y no habría nada que aprender.
- Con esta tabla, el precio que maximiza el ingreso de Ropa es el **70 %** del de referencia para los jóvenes, el **100 %** para los mayores y el **130 %** para los adultos. Todos son óptimos interiores (prueba `test_segments_have_different_optimal_prices`).

**Limitación del dataset:** `Price per Unit` solo toma 5 valores (25, 30, 50, 300, 500), en dos gamas. El precio de referencia (la media, ~180) representa un producto "promedio" de la categoría, no un artículo real.

**Números aleatorios comunes:** todo lo aleatorio de un día (segmento, número de clientes, disposición a pagar, cantidades) depende solo de la semilla, **no** del precio. Con la misma semilla, todas las políticas enfrentan exactamente los mismos clientes, y la comparación es pareada (prueba `test_same_seed_same_customers_for_any_policy`).

## 3. Entorno Gymnasium (`env.py`)

- Un **paso = un día**; un **episodio = 60 días**. El episodio termina por truncamiento (`truncated`), no por un estado terminal.
- **Estado** (7 números con un producto; 15 con tres):
  - precio actual: nivel / 10;
  - segmento de hoy (one-hot de 3);
  - ventas de los últimos 3 días (unidades / unidades de referencia).
- **Acciones discretas** (sirven para DQN y PPO): bajar 20 %, bajar 10 %, mantener, subir 10 %, subir 20 %. El precio vive en una grilla de 11 niveles (50 %–150 % del de referencia) y siempre se recorta a ese rango.
  - **Por qué ±2 niveles:** con solo ±1 (10 %), pasar del óptimo de los jóvenes (70 %) al de los adultos (130 %) tomaría 6 días, y un segmento dura en promedio entre 8 y 12 días: el agente pasaría buena parte de cada segmento en camino. Con ±2 toma 3.
  - Con tres productos, la acción es `MultiDiscrete([5, 5, 5])`. **DQN no admite MultiDiscrete**, por eso el multiproducto se entrena solo con PPO. Aplanar a `Discrete(125)` habría sido posible, pero no hace falta.
- **Recompensa** = ingreso del día (Σ precio × unidades) / 1000. Dividir por una constante no cambia la política óptima y deja los números en un rango cómodo para la red. El ingreso sin escalar va en `info["revenue"]`.
- **Orden de un paso:** el agente ve el segmento de hoy → fija el precio → compran los clientes de hoy → mañana puede cambiar el segmento.
- **Precio inicial:** al entrenar es al azar, para que el agente aprenda a salir de cualquier precio. En evaluación y en la pantalla se fija con `reset(options={"initial_level": k})`.
- **`check_env` de SB3 pasa** en las dos versiones, y se ejecuta antes de cada entrenamiento (`train.py`).

## 4. Agentes (`train.py`)

| Agente | Algoritmo | Pasos | Red | Tiempo (CPU) | Archivo |
|---|---|---|---|---|---|
| Un producto | DQN | 200 000 | 64-64 | ~30 s | `models/dqn_pricing.zip` (104 KB) |
| Un producto | PPO | 200 000 | 64-64 | ~30 s | `models/ppo_pricing.zip` (145 KB) |
| Tres productos | PPO | 400 000 | 128-128 | ~105 s | `models/ppo_pricing_multi.zip` (491 KB) |

- **γ = 0,9:** pesa más el ingreso de hoy, pero el agente mira unos 10 días adelante, lo que tarda en mover el precio al óptimo de otro segmento.
- **DQN:** buffer de 50 000, ε de 1 a 0,02 en el primer 30 % de los pasos y red objetivo actualizada cada 1 000 pasos.
- **PPO:** 1 200 pasos por actualización (20 episodios), 10 épocas y coeficiente de entropía 0,01.
- **Semilla 42 en todo.** La configuración de la demanda se guarda en `models/pricing_config.json`, así evaluación y pantalla usan exactamente la misma.
- **Qué hace SB3 por dentro** (para la sustentación):
  - **DQN** es el Q-learning de clase con una red en lugar de la tabla Q, más un replay buffer y una red objetivo (como en `RL_Market_Example.pdf`);
  - **PPO** es un método actor-crítico que actualiza la política con un objetivo "recortado" para no cambiarla demasiado en cada paso.
- **Curvas de aprendizaje** (`figures/learning_curves.png`): los tres agentes superan al mejor precio fijo a los ~30 000 pasos (un producto) y a los ~100 000 (tres productos), y se estabilizan cerca del oráculo.

## 5. Evaluación justa (`evaluate.py`)

- **Episodios:** 200, con semillas 10000–10199, distintas de las de entrenamiento.
- **Mismas semillas para todas las políticas:** gracias a los números aleatorios comunes, cada política ve los mismos clientes.
- **Referencia = el mejor precio fijo:** se prueban los 11 niveles y se elige el mejor en esos mismos 200 episodios, lo que lo favorece. Para Ropa resultó el 100 % ($174,29).
- **Mismo arranque:** el agente empieza en el mismo precio que el mejor precio fijo.
- **Estadística:** media, desviación, IC 95 % bootstrap de la diferencia **pareada** y % de episodios ganados.
- **Políticas de referencia:**
  - **oráculo:** conoce la demanda real y va al óptimo de cada segmento con los mismos movimientos permitidos. Es un **techo**, no una política alcanzable;
  - **precio al azar:** control.

### 5.1 Un producto (Ropa) — `comparison_single.csv`

| Política | Ingreso medio (60 días) | vs. mejor fijo | IC 95 % de la diferencia | Gana en |
|---|---|---|---|---|
| Mejor precio fijo ($174) | $779 168 | — | — | — |
| **DQN** | $871 789 | **+11,9 %** | [+89 252, +96 151] | 100 % |
| **PPO** | $862 659 | **+10,7 %** | [+80 012, +86 734] | 100 % |
| Oráculo (techo) | $875 724 | +12,4 % | [+92 751, +100 188] | 100 % |
| Precio al azar | $688 692 | −11,6 % | [−99 792, −81 309] | 6 % |

- **DQN y PPO capturan el 96 % y el 87 % de lo que es posible ganar** sobre el precio fijo: (871 789 − 779 168) / (875 724 − 779 168) = 0,96 para DQN.
- En este problema DQN rindió un poco más que PPO, con los dos entrenados el mismo número de pasos. **No se afirma que DQN sea mejor en general:** es el resultado de este entorno y estos hiperparámetros.
- **El agente aprendió a diferenciar por segmento** (`policy_by_segment.csv`, precio medio × referencia):

| Segmento | Óptimo | Oráculo | PPO | DQN |
|---|---|---|---|---|
| Jóvenes 18-35 | 0,70 | 0,75 | 0,73 | 0,81 |
| Mayores 56-64 | 1,00 | 1,00 | 0,97 | 0,98 |
| Adultos 36-55 | 1,30 | 1,27 | 1,23 | 1,19 |

- **El precio inicial casi no importa** (`initial_price_single.csv`): empezando en el 50 % o en el 150 %, PPO obtiene $858 812 y $857 036 (frente a $862 659 empezando en el 100 %). El agente corrige el precio en los primeros días. Un precio fijo en esos extremos daría solo $523 270 y $602 651.

### 5.2 Tres productos (bonus) — `comparison_multi.csv`

Mejores precios fijos: Ropa 100 %, Belleza 90 %, Electrónica 100 %.

| Política | Ingreso medio | vs. mejor fijo | IC 95 % | Gana en |
|---|---|---|---|---|
| Mejor precio fijo | $2 476 762 | — | — | — |
| **PPO** | $2 659 450 | **+7,4 %** | [+176 107, +189 434] | 100 % |
| Oráculo (techo) | $2 682 859 | +8,3 % | [+198 980, +213 362] | 100 % |
| Precio al azar | $2 137 595 | −13,7 % | — | 0 % |

La ganancia posible es menor que en Ropa (techo +8,3 % frente a +12,4 %). En Belleza y Electrónica los óptimos de los segmentos están más cerca del mejor precio fijo, así que hay menos que ganar ajustando.

## 6. Pantalla (`screen.py`)

- **Tres políticas viven el mismo episodio en paralelo**, con la misma semilla y los mismos clientes: el agente (PPO o DQN), el mejor precio fijo y el oráculo.
- **Gráfica de ingreso acumulado en vivo**, con etiquetas directas. El titular dice el % de ventaja del agente sobre el precio fijo.
- **Gráfica del precio** del agente día a día, con el óptimo del segmento (línea punteada), el mejor precio fijo y una franja con el segmento de cada día.
- **Controles:**
  - agente PPO / DQN;
  - "Solo ropa" / "Tres productos";
  - slider de **precio inicial** (11 niveles; al soltar se reinicia el entorno);
  - slider de velocidad (1–20 días/s);
  - **Reiniciar**, Pausa (o barra espaciadora) y "Otra semilla".
- Tooltip por día y nota con el resultado de los 200 episodios, para no sacar conclusiones de un solo episodio.
- **Solo carga** los agentes de `models/`: nunca entrena. Si faltan, muestra cómo entrenarlos en vez de romper el menú.

## 7. Limitaciones y qué decir en la sustentación

- La reacción al precio es **simulada**. Los números absolutos (+11 %) dependen de los supuestos de la tabla de disposición a pagar. Lo robusto es la conclusión cualitativa: si los segmentos tienen óptimos distintos y el segmento es observable, un agente de RL aprende a seguirlos y supera al mejor precio fijo.
- El agente **observa** el segmento del día. Si no lo observara, tendría que inferirlo de las ventas recientes, y la ganancia sería menor.
- Cerca del óptimo, la curva de ingreso es plana. Por eso PPO a veces alterna entre dos precios vecinos (110 %–140 % con los adultos, `episode_example.png`) sin perder casi nada.
- No hay inventario, costos, competencia ni efecto de "precio de referencia" en la memoria del cliente. La recompensa es ingreso, no utilidad, como pide el enunciado.
- **Reproducibilidad:** `python -m task_c_pricing.train` y luego `python -m task_c_pricing.evaluate`, con semilla 42 y en CPU.
