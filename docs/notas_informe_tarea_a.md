# Tarea A — Rutas de clientes y layout (BFS y A*): notas para el informe

> Documento de trabajo con supuestos, decisiones (con números) y resultados reales de la Tarea A.
> Todo número sale de `python -m task_a_pathfinding.evaluate` (archivos en `outputs/task_a/`).

## 1. Requisitos (enunciado `UrbanStyle_Project_Plan-2026_2.pdf`)

**Objetivo:** mostrar cómo se mueven los clientes dentro de la tienda e identificar los puntos congestionados.

| # | Obligatorio | Dónde se cumple |
|---|---|---|
| 1 | Tienda como grilla (p. ej. 8×6) | `store.py`: grilla de 16 × 10 celdas (§2) |
| 2 | Mapear categorías de producto a secciones | `store.py::SECTIONS`, `section_for` |
| 3 | Clientes que entran y se mueven a las secciones | `simulation.py` |
| 4 | Al menos BFS y A* | `search.py` |
| 5 | Visualizar zonas de alto tráfico | `simulation.py::high_traffic_mask` + pantalla + `heatmaps.png` |
| Pygame | Grilla, clientes en movimiento, sliders/botones para el algoritmo o el número de clientes | `screen.py` |
| Bonus | Mapa de calor superpuesto; probar distintos layouts | capa "Calor" de la pantalla; 3 layouts y `layouts.csv` |

Rúbrica: 15 %.

## 2. Modelo de la tienda

- **Grilla de 16 × 10** (el enunciado dice "p. ej. 8×6"). Se usó una más grande porque en 8×6, quitando paredes, quedan muy pocas celdas: no se forman pasillos alternativos y BFS y A* casi no se diferencian.
- Cada layout es un mapa de texto (`#` estantería/pared, `.` pasillo, `E` entrada, `S` salida, `C` caja, letras de sección). La grilla es un **grafo implícito**: los vecinos de una celda se calculan al vuelo (4 direcciones, costo 1 por paso).
- **Secciones:** el dataset tiene 3 categorías. La ropa se divide por género porque en una tienda de moda se exhibe en zonas separadas:

| Sección | Categoría (dataset) | Género |
|---|---|---|
| Ropa mujer (F) | Clothing | Female |
| Ropa hombre (M) | Clothing | Male |
| Belleza (B) | Beauty | cualquiera |
| Electrónica (T) | Electronics | cualquiera |

- **Validación:** `Store.validate()` comprueba que haya una entrada y una salida, una caja, que **todas** las celdas de sección, caja y salida sean alcanzables desde la entrada y que no haya celdas transitables aisladas. Los 3 layouts pasan (prueba `test_layouts_are_valid_and_have_every_section`).

### 2.1 Los tres layouts (bonus)

Las tres distribuciones tienen el mismo tamaño y la misma superficie por sección (8, 8, 6 y 6 celdas). Cambia solo la ubicación de los pasillos.

| Layout | Idea | Celdas transitables |
|---|---|---|
| Pasillos | Góndolas en paralelo: varios caminos alternativos | 90 |
| Boutique | Isla central de exhibición con un anillo de circulación | 90 |
| Recorrido | Recorrido guiado (tipo IKEA): desde la entrada hay un solo pasillo | 78 |

## 3. Algoritmos de búsqueda (`search.py`)

- **BFS:** frontera FIFO (`deque`). Con costo uniforme encuentra la ruta más corta.
- **A\*:** frontera con `heapq` ordenada por `f(n) = g(n) + h(n)`, con heurística **Manhattan**. Es admisible y consistente en una grilla de 4 direcciones (nunca sobreestima; prueba `test_manhattan_is_admissible_on_layout`), así que A* también es óptimo. Empates: menor `h` primero y luego orden de llegada, para que sea determinista.
- **Comparación en las mismas condiciones:**
  - mismo orden de vecinos;
  - prueba de meta al **expandir** en los dos;
  - "nodos expandidos" = nodos sacados de la frontera, incluida la meta.
- Si no hay ruta se devuelve `path = None`, y la simulación salta esa parada sin romperse (prueba `test_unreachable_goal_returns_none`).

### 3.1 Resultados BFS vs. A* (`metrics/algorithm_summary.csv`)

Todos los pares (origen, destino) de puntos clave: 992 por layout, 2 976 en total.

| Layout | Mismo largo | Largo medio | Nodos BFS | Nodos A* | A*/BFS | A* expande menos | Tiempo BFS | Tiempo A* |
|---|---|---|---|---|---|---|---|---|
| Pasillos | 100 % | 9,5 | 48,5 | 12,1 | 0,25 | 97,5 % | 38,6 µs | 14,7 µs |
| Boutique | 100 % | 9,0 | 46,2 | 10,5 | 0,23 | 97,7 % | 37,1 µs | 13,2 µs |
| Recorrido | 100 % | 14,1 | 43,0 | 23,6 | 0,55 | 97,3 % | 36,0 µs | 27,7 µs |

**Lectura:**
- **Las rutas tienen siempre el mismo largo** (100 % de los pares): con costo uniforme y una heurística admisible, A* es óptimo igual que BFS.
- A* expande entre el 23 % y el 55 % de los nodos de BFS. En el 2–3 % restante de los pares expande **lo mismo**, nunca más; son casos con el destino pegado al origen.
- **En "Recorrido" A\* ahorra mucho menos** (0,55): la distancia Manhattan subestima mucho el camino real, que obliga a dar vueltas, así que la heurística orienta poco. **No se da por hecho que A\* siempre gana por mucho:** depende de qué tan buena sea la heurística en ese layout.
- Los tiempos (microsegundos) son de Python y dependen de la máquina; los nodos expandidos son la medida que no depende del hardware.
- `figures/exploration.png` lo muestra en un mismo trayecto (entrada → Ropa hombre): BFS expande 78 nodos y A* 19, y los dos encuentran una ruta de 18 pasos.

## 4. Simulación (`simulation.py`)

- **Clientes = transacciones reales:** cada cliente se toma al azar del CSV (categoría, género y cantidad). Así, la proporción de clientes por sección es la real del dataset.
- **Recorrido:** entrada → una celda al azar de su sección → una caja → salida. Cada tramo se planea con el algoritmo elegido, así que hay 3 búsquedas por cliente.
- **Tiempos (supuestos):**
  - llegadas al azar en una ventana de 30 s; más clientes en la misma ventana generan más congestión;
  - velocidad entre 2,5 y 3,5 celdas/s;
  - tiempo mirando en la sección: 1,5 s + 0,75 s por unidad comprada;
  - tiempo en la caja: 1 s + 0,25 s por unidad.
- **La velocidad no depende de los FPS:** la simulación avanza con `step(dt)` en segundos simulados. La prueba `test_simulation_is_deterministic_and_fps_independent` da el mismo tráfico con `dt = 0,05` y con `dt = 1/13`.
- **Decisión (pendiente del plan §5.1):** varios clientes **pueden** ocupar la misma celda. La congestión se **mide** (pico simultáneo) en vez de bloquearla. Bloquear celdas obligaría a re-planear en tiempo real, y entonces BFS y A* ya no serían comparables.

### 4.1 Medida objetiva del tráfico

| Medida | Qué es |
|---|---|
| `visits` | Veces que un cliente **entra** a la celda (medida principal del mapa de calor). La suma de visitas es igual a la suma de los largos de las rutas (prueba) |
| `occupancy` | Segundos-cliente en la celda (incluye mirar y pagar) |
| `peak` | Máximo de clientes a la vez en la celda |
| Zona de alto tráfico | Celdas con visitas en el **10 % superior** de las celdas transitadas (`high_traffic_mask`) |

## 5. Experimento de layouts (`metrics/layouts.csv`)

100 clientes, A*, mismas 10 semillas en los 3 layouts:

| Layout | Ruta media (pasos) | Tiempo en tienda | Pico en una celda | Visitas en el top-5 | Pasillo más transitado |
|---|---|---|---|---|---|
| Pasillos | 28,3 | 14,6 s | 9,4 | 24,9 % | junto a la caja (146 visitas) |
| Boutique | 30,6 | 15,4 s | 9,7 | 24,0 % | junto a la caja (152) |
| Recorrido | 42,5 | 19,4 s | 10,6 | 18,2 % | junto a la caja (200) |

**Conclusiones para UrbanStyle:**
- En los tres layouts, **el punto más congestionado es el pasillo entre la caja y la salida**. Todo cliente pasa por él dos veces: al ir a pagar y al salir. Recomendación: separar la fila de la caja del camino a la salida, o poner una segunda caja cerca de la entrada.
- **Pasillos** da las rutas más cortas (28,3 pasos) y el menor tiempo en tienda.
- **Recorrido** alarga las rutas un 50 % (más exposición a producto, que es la idea de ese formato), pero concentra el tráfico en un solo pasillo: el pasillo junto a la caja recibe 200 visitas y el pico sube a 10,6 clientes. Reparte más el tráfico entre celdas (top-5 = 18 %) porque los obliga a recorrer más tienda.

## 6. Pantalla (`screen.py`)

- **Controles:**
  - **algoritmo** BFS / A*: se re-simula con **los mismos clientes** (misma semilla), para compararlos de forma justa;
  - **distribución** (3 layouts);
  - slider de **número de clientes** (5–100, se aplica al soltar);
  - slider de **velocidad** (0,5×–4×);
  - Reiniciar, Pausa (o barra espaciadora) y "Otra muestra" (otra semilla).
- **Capas:**
  - **Calor:** mapa de calor de visitas y contorno en las zonas de alto tráfico;
  - **Rutas:** lo que le falta por recorrer a cada cliente;
  - **Búsqueda:** las celdas que expandió el algoritmo en el tramo entrada → sección del último cliente que entró, y su ruta. Con BFS se ve cómo "inunda" la tienda; con A*, que va casi directo.
- El titular nombra en vivo el pasillo más transitado. El panel muestra atendidos, ruta media, nodos expandidos, tiempo por búsqueda, celda más visitada y pico.
- El tooltip de cada celda muestra sus visitas, ocupación y pico.

## 7. Limitaciones

- Los tiempos de caminata y de compra son supuestos. El dataset no tiene datos de movimiento en tienda, solo qué se compró.
- Cada cliente visita **una** sección, porque cada transacción del dataset es de una sola categoría. En una tienda real un cliente visita varias.
- Sin bloqueo entre clientes, la congestión se mide pero no frena a nadie: en hora pico los tiempos reales serían mayores.
- La heurística es Manhattan: es la adecuada para 4 direcciones; con movimiento diagonal habría que usar otra.
