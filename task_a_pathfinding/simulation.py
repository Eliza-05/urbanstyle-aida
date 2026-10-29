"""Tarea A (Sebastián): simulación de clientes recorriendo la tienda y conteo objetivo del tráfico.

Cada cliente es una transacción REAL del dataset (categoría, género y cantidad): entra, camina a la sección
de su compra, se queda mirando un tiempo que crece con la cantidad que compra, paga en la caja y sale.
Cada tramo (entrada -> sección -> caja -> salida) se planea con el algoritmo elegido (BFS o A*).

Medidas de tráfico (todas por celda):
  * visits:    cuántas veces un cliente ENTRA a la celda (la medida principal del mapa de calor);
  * occupancy: segundos-cliente que pasa gente en la celda (incluye el tiempo mirando o pagando);
  * peak:      máximo de clientes a la vez en la celda (congestión).
Decisión: varios clientes pueden ocupar la misma celda (no se bloquean); la congestión se MIDE con `peak`
en lugar de impedirla, así las rutas de BFS y A* siguen siendo comparables.

La simulación avanza con `step(dt)` en segundos simulados: la velocidad no depende de los FPS.
"""
from dataclasses import dataclass, field

import numpy as np

from common import config
from task_a_pathfinding.search import ALGORITHMS, SearchResult
from task_a_pathfinding.store import SECTIONS, Cell, Store, section_for

ARRIVAL_WINDOW = 30.0          # s: los clientes llegan repartidos al azar en esta ventana
SPEED_RANGE = (2.5, 3.5)       # celdas por segundo
BROWSE_BASE, BROWSE_PER_ITEM = 1.5, 0.75   # s en la sección: base + por unidad comprada
PAY_BASE, PAY_PER_ITEM = 1.0, 0.25         # s en la caja
HEADLESS_DT = 0.05             # paso fijo para simular sin pantalla


@dataclass(frozen=True)
class Profile:
    """Lo que el cliente compra: sección (letra) y unidades."""
    section: str
    quantity: int


def profiles_from_sales(df) -> list[Profile]:
    """Un perfil por transacción del dataset (así las proporciones por sección son las reales)."""
    return [Profile(section_for(cat, gen), int(q))
            for cat, gen, q in zip(df["Product Category"], df["Gender"], df["Quantity"])]


def load_profiles() -> list[Profile]:
    """Perfiles del CSV de ventas; si no está, uno por sección y cantidad (todas las secciones por igual)."""
    try:
        from common.data import load_sales
        return profiles_from_sales(load_sales())
    except (FileNotFoundError, ValueError):
        return [Profile(s, q) for s in SECTIONS for q in range(1, 5)]


@dataclass
class Customer:
    id: int
    profile: Profile
    arrival: float                  # s simulados
    speed: float                    # celdas/s
    path: list[Cell] = field(default_factory=list)
    dwell_at: dict[int, float] = field(default_factory=dict)  # índice en path -> segundos de parada
    pos: float = 0.0                # posición continua a lo largo de path
    dwell_left: float = 0.0
    state: str = "waiting"          # waiting -> walking/browsing/paying -> done
    left: float | None = None       # s simulados en que salió de la tienda

    @property
    def active(self) -> bool:
        return self.state not in ("waiting", "done")

    @property
    def cell(self) -> Cell:
        return self.path[min(int(round(self.pos)), len(self.path) - 1)]

    def xy(self) -> tuple[float, float]:
        """Posición continua (fila, columna) para dibujar el movimiento suave entre celdas."""
        i = int(self.pos)
        if i >= len(self.path) - 1:
            return tuple(map(float, self.path[-1]))
        t = self.pos - i
        (r0, c0), (r1, c1) = self.path[i], self.path[i + 1]
        return r0 + (r1 - r0) * t, c0 + (c1 - c0) * t


class Simulation:
    """Clientes que entran, recorren la tienda y salen. Métodos: step(dt), run_to_end(), stats()."""

    def __init__(self, store: Store, algorithm: str = "A*", n_customers: int = 30, seed: int = config.SEED,
                 profiles: list[Profile] | None = None, arrival_window: float = ARRIVAL_WINDOW):
        if algorithm not in ALGORITHMS:
            raise KeyError(f"Algoritmo desconocido: {algorithm!r}. Opciones: {list(ALGORITHMS)}")
        self.store, self.algorithm = store, algorithm
        self.search = ALGORITHMS[algorithm]
        self.rng = np.random.default_rng(seed)
        profiles = profiles or load_profiles()
        arrivals = np.sort(self.rng.uniform(0, arrival_window, n_customers))
        picks = self.rng.integers(0, len(profiles), n_customers)
        speeds = self.rng.uniform(*SPEED_RANGE, n_customers)
        self.customers = [Customer(i, profiles[p], float(t), float(v))
                          for i, (t, p, v) in enumerate(zip(arrivals, picks, speeds))]
        self.time = 0.0
        shape = (store.n_rows, store.n_cols)
        self.visits = np.zeros(shape, dtype=int)
        self.occupancy = np.zeros(shape)
        self.peak = np.zeros(shape, dtype=int)
        self.searches: list[SearchResult] = []
        self.last_search: tuple[Cell, Cell, SearchResult] | None = None  # para dibujar la exploración
        self.no_route = 0

    # ---------- planeación ----------
    def plan(self, customer: Customer) -> None:
        """Planea los 3 tramos del cliente con el algoritmo elegido y marca dónde se detiene."""
        store, q = self.store, customer.profile.quantity
        section_cells = store.sections.get(customer.profile.section)
        if not section_cells:  # el layout no tiene esa sección: va directo a pagar
            section_cells = store.checkouts
        stops = [
            (section_cells[self.rng.integers(len(section_cells))], BROWSE_BASE + BROWSE_PER_ITEM * q),
            (store.checkouts[self.rng.integers(len(store.checkouts))], PAY_BASE + PAY_PER_ITEM * q),
            (store.exit, 0.0),
        ]
        path, here = [store.entrance], store.entrance
        for leg, (goal, dwell) in enumerate(stops):
            result = self.search(here, goal, store.neighbors)
            self.searches.append(result)
            if leg == 0:  # se dibuja el tramo entrada -> sección: el más largo y el que más distingue BFS de A*
                self.last_search = (here, goal, result)
            if not result.found:  # destino sin ruta: se salta esa parada (no rompe la simulación)
                self.no_route += 1
                continue
            path += result.path[1:]
            here = goal
            if dwell:
                customer.dwell_at[len(path) - 1] = dwell
        customer.path = path

    # ---------- avance ----------
    def _enter(self, customer: Customer, index: int) -> None:
        self.visits[customer.path[index]] += 1
        if index in customer.dwell_at:
            customer.dwell_left = customer.dwell_at.pop(index)
            customer.state = "paying" if customer.path[index] in self.store.checkouts else "browsing"
        elif index == len(customer.path) - 1:
            customer.state = "done"

    def _advance(self, customer: Customer, dt: float) -> None:
        while dt > 1e-12 and customer.state != "done":
            if customer.dwell_left > 0:
                used = min(dt, customer.dwell_left)
                customer.dwell_left -= used
                dt -= used
                if customer.dwell_left <= 1e-12:
                    customer.dwell_left = 0.0
                    customer.state = "done" if customer.pos >= len(customer.path) - 1 else "walking"
                continue
            nxt = int(customer.pos) + 1
            need = (nxt - customer.pos) / customer.speed
            if dt >= need:
                customer.pos = float(nxt)
                dt -= need
                self._enter(customer, nxt)
            else:
                customer.pos += customer.speed * dt
                dt = 0.0

    def step(self, dt: float) -> None:
        """Avanza `dt` segundos simulados."""
        end = self.time + dt
        for c in self.customers:
            if c.state == "waiting" and c.arrival <= end:
                self.plan(c)
                c.state = "walking"
                self._enter(c, 0)
                self._advance(c, end - c.arrival)
            elif c.active:
                before = c.cell
                self._advance(c, dt)
                self.occupancy[before] += dt
            if c.state == "done" and c.left is None:
                c.left = end
        self.time = end
        here = np.zeros_like(self.peak)
        for c in self.customers:
            if c.active:
                here[c.cell] += 1
        np.maximum(self.peak, here, out=self.peak)

    @property
    def finished(self) -> bool:
        return all(c.state == "done" for c in self.customers)

    def run_to_end(self, dt: float = HEADLESS_DT, max_time: float = 600.0) -> "Simulation":
        while not self.finished and self.time < max_time:
            self.step(dt)
        return self

    # ---------- resultados ----------
    def hotspots(self, k: int = 5, aisles_only: bool = False) -> list[tuple[Cell, int]]:
        """Las k celdas con más visitas (opcionalmente solo pasillos), de mayor a menor."""
        cells = [c for c in self.store.walkable if not aisles_only or self.store.kind(c) == "."]
        ranked = sorted(cells, key=lambda c: (-self.visits[c], c))
        return [(c, int(self.visits[c])) for c in ranked[:k] if self.visits[c] > 0]

    def high_traffic_mask(self, quantile: float = 0.9) -> np.ndarray:
        """Zonas de alto tráfico: celdas con visitas en el 10 % superior de las celdas transitadas."""
        used = self.visits[self.visits > 0]
        if used.size == 0:
            return np.zeros_like(self.visits, dtype=bool)
        return self.visits >= max(1, np.quantile(used, quantile))

    def stats(self) -> dict:
        lengths = [len(c.path) - 1 for c in self.customers if c.path]
        found = [s for s in self.searches if s.found]
        stays = [c.left - c.arrival for c in self.customers if c.left is not None]
        return {
            "customers": len(self.customers),
            "arrived": sum(c.state != "waiting" for c in self.customers),
            "in_store": sum(c.active for c in self.customers),
            "served": sum(c.state == "done" for c in self.customers),
            "searches": len(self.searches),
            "no_route": self.no_route,
            "mean_route": float(np.mean(lengths)) if lengths else 0.0,
            "mean_time_in_store": float(np.mean(stays)) if stays else 0.0,
            "mean_expanded": float(np.mean([s.expanded for s in self.searches])) if self.searches else 0.0,
            "mean_search_us": float(np.mean([s.seconds for s in found]) * 1e6) if found else 0.0,
            "max_peak": int(self.peak.max()),
            "time": self.time,
        }
