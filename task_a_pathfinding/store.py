"""Tarea A (Sebastián): la tienda como grilla (grafo implícito), sus secciones y las distribuciones a comparar.

Cada distribución se escribe como un mapa de texto de 16 × 10 celdas:

    #  estantería o pared (no se puede pisar)     .  pasillo
    E  entrada                                     S  salida
    C  caja
    F  Ropa mujer   (Clothing + Female)           M  Ropa hombre (Clothing + Male)
    B  Belleza      (Beauty)                       T  Electrónica (Electronics)

Las secciones (F, M, B, T) y la caja son celdas transitables: el cliente camina HASTA ellas. Las categorías
del dataset son 3 (Clothing, Beauty, Electronics); la ropa se divide por género porque en una tienda de moda
se exhibe en zonas separadas. El movimiento es en 4 direcciones y cada paso cuesta 1.
"""
from dataclasses import dataclass
from functools import cached_property

Cell = tuple[int, int]  # (fila, columna)

WALL, FLOOR, ENTRANCE, EXIT, CHECKOUT = "#", ".", "E", "S", "C"
SECTIONS = {  # letra -> (nombre visible, categoría del dataset, género o None si no depende del género)
    "F": ("Ropa mujer", "Clothing", "Female"),
    "M": ("Ropa hombre", "Clothing", "Male"),
    "B": ("Belleza", "Beauty", None),
    "T": ("Electrónica", "Electronics", None),
}
CELL_NAMES = {WALL: "Estantería", FLOOR: "Pasillo", ENTRANCE: "Entrada", EXIT: "Salida", CHECKOUT: "Caja",
              **{k: v[0] for k, v in SECTIONS.items()}}
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # arriba, derecha, abajo, izquierda (orden fijo para BFS y A*)

# Tres distribuciones (bonus "probar distintos layouts"). Todas tienen 16 × 10 celdas y la misma superficie
# por sección, para que la comparación dependa solo de cómo se ubican los pasillos.
LAYOUTS = {
    # Góndolas en paralelo: varios pasillos alternativos entre la entrada y cada sección.
    "Pasillos": (
        "################",
        "#FFFF......MMMM#",
        "#FFFF.##.##MMMM#",
        "#.....##.##....#",
        "#.##.......##..#",
        "#.##.##.##.##..#",
        "#BBB.##.##..TTT#",
        "#BBB........TTT#",
        "#.....CC.......#",
        "#E############S#",
    ),
    # Isla central de exhibición: un anillo de circulación alrededor de un bloque.
    "Boutique": (
        "################",
        "#FFFF......MMMM#",
        "#FFFF......MMMM#",
        "#...########...#",
        "#BB.########.TT#",
        "#BB.########.TT#",
        "#BB..........TT#",
        "#.....CC.......#",
        "#..............#",
        "#E############S#",
    ),
    # Recorrido guiado (tipo IKEA): desde la entrada hay un solo pasillo, que sube por la izquierda y cruza
    # por arriba antes de bajar a las demás secciones, la caja y la salida.
    "Recorrido": (
        "################",
        "#FFFF.....#MMMM#",
        "#FFFF.###.#MMMM#",
        "#.....###.#....#",
        "#.#######.####.#",
        "#.#BB.....#TT..#",
        "#.#BB.###.#TT..#",
        "#.#BB.###..TT..#",
        "#.#####CC......#",
        "#E############S#",
    ),
}
DEFAULT_LAYOUT = "Pasillos"


@dataclass(frozen=True)
class Store:
    """Grilla de la tienda. Las celdas son (fila, columna); los vecinos se calculan al vuelo (grafo implícito)."""

    name: str
    rows_text: tuple[str, ...]

    def __post_init__(self):
        widths = {len(row) for row in self.rows_text}
        if len(widths) != 1:
            raise ValueError(f"El layout {self.name!r} no es rectangular: anchos {sorted(widths)}")
        unknown = set("".join(self.rows_text)) - set(CELL_NAMES)
        if unknown:
            raise ValueError(f"Símbolos desconocidos en {self.name!r}: {sorted(unknown)}")

    @property
    def n_rows(self) -> int:
        return len(self.rows_text)

    @property
    def n_cols(self) -> int:
        return len(self.rows_text[0])

    def kind(self, cell: Cell) -> str:
        r, c = cell
        return self.rows_text[r][c]

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.n_rows and 0 <= cell[1] < self.n_cols

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.kind(cell) != WALL

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Celdas transitables a un paso (4 direcciones, siempre en el orden de MOVES)."""
        r, c = cell
        return [(r + dr, c + dc) for dr, dc in MOVES if self.passable((r + dr, c + dc))]

    def cells_of(self, kind: str) -> list[Cell]:
        return [(r, c) for r, row in enumerate(self.rows_text) for c, ch in enumerate(row) if ch == kind]

    @cached_property
    def walkable(self) -> list[Cell]:
        return [(r, c) for r in range(self.n_rows) for c in range(self.n_cols) if self.passable((r, c))]

    @cached_property
    def entrance(self) -> Cell:
        return self.cells_of(ENTRANCE)[0]

    @cached_property
    def exit(self) -> Cell:
        return self.cells_of(EXIT)[0]

    @cached_property
    def checkouts(self) -> list[Cell]:
        return self.cells_of(CHECKOUT)

    @cached_property
    def sections(self) -> dict[str, list[Cell]]:
        """Letra de sección -> sus celdas (solo las secciones presentes en el layout)."""
        return {k: self.cells_of(k) for k in SECTIONS if self.cells_of(k)}

    def reachable_from(self, start: Cell) -> set[Cell]:
        """Componente conexa de `start` (recorrido en anchura sin destino)."""
        seen, frontier = {start}, [start]
        while frontier:
            nxt = []
            for cell in frontier:
                for n in self.neighbors(cell):
                    if n not in seen:
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt
        return seen

    def validate(self) -> list[str]:
        """Problemas del layout (lista vacía = válido): una entrada, una salida, caja y TODAS las secciones
        alcanzables desde la entrada, y ninguna celda transitable aislada."""
        problems = []
        for kind, label in ((ENTRANCE, "entrada"), (EXIT, "salida")):
            if len(self.cells_of(kind)) != 1:
                problems.append(f"debe haber exactamente una {label}")
        if not self.checkouts:
            problems.append("no hay caja")
        if problems:
            return problems
        reach = self.reachable_from(self.entrance)
        targets = {"salida": [self.exit], "caja": self.checkouts,
                   **{CELL_NAMES[k]: cells for k, cells in self.sections.items()}}
        for label, cells in targets.items():
            missing = [c for c in cells if c not in reach]
            if missing:
                problems.append(f"{label}: {len(missing)} celda(s) sin ruta desde la entrada")
        isolated = [c for c in self.walkable if c not in reach]
        if isolated:
            problems.append(f"{len(isolated)} celda(s) transitables aisladas")
        return problems

    def describe(self, cell: Cell) -> str:
        """Nombre legible de una celda: la sección, o el pasillo junto a la sección/caja más cercana."""
        kind = self.kind(cell)
        if kind != FLOOR:
            return CELL_NAMES[kind]
        best = None
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                k = self.rows_text[r][c]
                if k in (FLOOR, WALL):
                    continue
                d = abs(r - cell[0]) + abs(c - cell[1])
                if best is None or d < best[0]:
                    best = (d, k)
        if best is not None and best[0] <= 2:
            return f"Pasillo junto a {CELL_NAMES[best[1]]}"
        return f"Pasillo (fila {cell[0] + 1}, col. {cell[1] + 1})"


def load_layout(name: str = DEFAULT_LAYOUT) -> Store:
    if name not in LAYOUTS:
        raise KeyError(f"Layout desconocido: {name!r}. Opciones: {list(LAYOUTS)}")
    return Store(name, LAYOUTS[name])


def section_for(category: str, gender: str) -> str:
    """Sección (letra) a la que va un cliente según la categoría y el género de su compra en el dataset."""
    for letter, (_, cat, gen) in SECTIONS.items():
        if cat == category and gen in (None, gender):
            return letter
    raise KeyError(f"Sin sección para {category!r} / {gender!r}")
