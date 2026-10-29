"""Tarea A (Sebastián): algoritmos de búsqueda BFS y A* sobre la grilla de la tienda.

Los dos reciben una función `neighbors(celda) -> [celdas]` (el grafo implícito de store.Store), así se
prueban con cualquier grilla. Para compararlos en las MISMAS condiciones:
  * mismo orden de vecinos (store.MOVES) y la prueba de meta se hace al EXPANDIR (sacar de la frontera);
  * "nodos expandidos" = nodos sacados de la frontera hasta encontrar la meta, incluida la meta;
  * costo uniforme (1 por paso). Con la heurística Manhattan, que es admisible y consistente en una grilla
    de 4 direcciones, A* encuentra rutas del mismo largo que BFS (las dos son óptimas).
Si no hay ruta, `path` es None (la simulación lo maneja sin romperse).
"""
import heapq
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

Cell = tuple[int, int]
Neighbors = Callable[[Cell], list[Cell]]


@dataclass
class SearchResult:
    path: list[Cell] | None         # de start a goal, ambos incluidos; None si no hay ruta
    expanded: int                   # nodos sacados de la frontera
    seconds: float                  # tiempo de la búsqueda
    explored: list[Cell] = field(default_factory=list)  # orden de expansión (para dibujar la exploración)

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> int | None:
        """Número de pasos de la ruta (celdas - 1)."""
        return None if self.path is None else len(self.path) - 1


def _rebuild(parent: dict, goal: Cell) -> list[Cell]:
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs(start: Cell, goal: Cell, neighbors: Neighbors) -> SearchResult:
    """Búsqueda en anchura: frontera FIFO. Con costo uniforme encuentra la ruta más corta."""
    t0 = time.perf_counter()
    parent = {start: None}
    frontier = deque([start])
    explored = []
    while frontier:
        cell = frontier.popleft()
        explored.append(cell)
        if cell == goal:
            return SearchResult(_rebuild(parent, goal), len(explored), time.perf_counter() - t0, explored)
        for n in neighbors(cell):
            if n not in parent:  # cada celda entra a la frontera una sola vez
                parent[n] = cell
                frontier.append(n)
    return SearchResult(None, len(explored), time.perf_counter() - t0, explored)


def astar(start: Cell, goal: Cell, neighbors: Neighbors,
          heuristic: Callable[[Cell, Cell], float] = manhattan) -> SearchResult:
    """A*: frontera ordenada por f(n) = g(n) + h(n). Empates: menor h primero (más cerca de la meta) y luego
    orden de llegada, para que el resultado sea determinista."""
    t0 = time.perf_counter()
    g = {start: 0}
    parent = {start: None}
    counter = 0
    frontier = [(heuristic(start, goal), heuristic(start, goal), counter, start)]
    closed = set()
    explored = []
    while frontier:
        _, _, _, cell = heapq.heappop(frontier)
        if cell in closed:  # entrada vieja de la cola (se encontró luego un camino mejor)
            continue
        closed.add(cell)
        explored.append(cell)
        if cell == goal:
            return SearchResult(_rebuild(parent, goal), len(explored), time.perf_counter() - t0, explored)
        for n in neighbors(cell):
            new_g = g[cell] + 1
            if n not in closed and new_g < g.get(n, float("inf")):
                g[n] = new_g
                parent[n] = cell
                counter += 1
                h = heuristic(n, goal)
                heapq.heappush(frontier, (new_g + h, h, counter, n))
    return SearchResult(None, len(explored), time.perf_counter() - t0, explored)


ALGORITHMS: dict[str, Callable[[Cell, Cell, Neighbors], SearchResult]] = {"BFS": bfs, "A*": astar}
