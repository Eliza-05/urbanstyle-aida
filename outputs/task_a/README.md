# Salidas de la Tarea A (rutas de clientes y layout)

Todo se regenera con `python -m task_a_pathfinding.evaluate` (~6 s). Unidades: celdas de la grilla (pasos) y visitas.

## metrics/ — BFS vs. A* y experimento de layouts
- `algorithm_pairs.csv`: una fila por par (origen, destino) de puntos clave de cada layout (entrada, celdas de sección, cajas y salida; 992 pares por layout). Para cada algoritmo: largo de la ruta (`*_length`), nodos expandidos (`*_expanded`) y tiempo mediano de 25 repeticiones en microsegundos (`*_us`).
- `algorithm_summary.csv`: resumen por layout: % de pares con el mismo largo en BFS y A* (`same_length_pct`, debe ser 100), nodos expandidos medios, % de pares en que A* expande menos (`astar_fewer_pct`) o igual (`astar_equal_pct`), tiempos y cociente A*/BFS de nodos expandidos.
- `layouts.csv`: los mismos clientes (10 semillas × 100 clientes, A*) en los 3 layouts: ruta media (pasos), tiempo en tienda (s), duración total, pico de clientes en una celda, % de visitas en las 5 celdas más visitadas (concentración) y pasillo más transitado.

## traffic/ — tráfico por celda (semilla 0, 100 clientes, A*)
- `traffic_<layout>.csv`: `row`, `col`, tipo de celda, descripción legible, `visits` (entradas a la celda), `occupancy_s` (segundos-cliente en la celda) y `peak` (máximo de clientes a la vez).

## figures/
- `heatmaps.png`: mapa de calor de visitas medias por celda en cada layout (bonus).
- `exploration.png`: un mismo trayecto (entrada → Ropa hombre): celdas que expande BFS y A* y la ruta (18 pasos en los dos).
- `expanded.png`: nodos expandidos medios por layout y algoritmo.
- `screen_task_a.png`, `screen_task_a_busqueda.png`: capturas de la pantalla (mapa de calor; capa "Búsqueda" con BFS).
