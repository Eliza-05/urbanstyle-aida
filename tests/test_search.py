"""Pruebas de la Tarea A: grilla, BFS y A*, simulación y pantalla (rápidas; no necesitan el CSV)."""
import itertools

import numpy as np
import pygame
import pytest

from task_a_pathfinding.search import ALGORITHMS, astar, bfs, manhattan
from task_a_pathfinding.simulation import Profile, Simulation
from task_a_pathfinding.store import LAYOUTS, SECTIONS, Store, load_layout, section_for

PROFILES = [Profile(s, q) for s in SECTIONS for q in range(1, 5)]  # sin depender del CSV


def grid_neighbors(rows: list[str]):
    """Vecinos en 4 direcciones de una grilla de texto ('#' = obstáculo)."""
    def neighbors(cell):
        r, c = cell
        out = []
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < len(rows) and 0 <= cc < len(rows[0]) and rows[rr][cc] != "#":
                out.append((rr, cc))
        return out
    return neighbors


def assert_valid_path(path, start, goal, neighbors):
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert b in neighbors(a)  # pasos contiguos y transitables


# ---------- búsqueda ----------

@pytest.mark.parametrize("algo", [bfs, astar])
def test_shortest_path_with_obstacle(algo):
    rows = ["....",
            ".##.",
            "...."]
    nb = grid_neighbors(rows)
    result = algo((1, 0), (1, 3), nb)
    assert result.length == 5  # tiene que rodear el bloque: 1 + 3 + 1
    assert_valid_path(result.path, (1, 0), (1, 3), nb)


@pytest.mark.parametrize("algo", [bfs, astar])
def test_unreachable_goal_returns_none(algo):
    rows = ["..#.",
            "..#.",
            "..#."]
    result = algo((0, 0), (0, 3), grid_neighbors(rows))
    assert result.path is None and result.length is None and not result.found
    assert result.expanded == 6  # exploró todo lo alcanzable antes de rendirse


@pytest.mark.parametrize("algo", [bfs, astar])
def test_start_equals_goal(algo):
    result = algo((0, 0), (0, 0), grid_neighbors(["..", ".."]))
    assert result.path == [(0, 0)] and result.length == 0 and result.expanded == 1


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_bfs_and_astar_same_length_on_every_layout(name):
    """Costo uniforme + Manhattan admisible: A* es óptimo, igual que BFS (en TODOS los pares)."""
    store = load_layout(name)
    points = [store.entrance, store.exit, *store.checkouts, *[c for cs in store.sections.values() for c in cs]]
    for a, b in itertools.permutations(points, 2):
        rb, ra = bfs(a, b, store.neighbors), astar(a, b, store.neighbors)
        assert rb.length == ra.length
        assert_valid_path(ra.path, a, b, store.neighbors)
        assert ra.expanded <= rb.expanded  # en estas grillas A* nunca expande más que BFS


def test_astar_expands_fewer_nodes_in_open_grid():
    nb = grid_neighbors(["." * 12] * 12)
    assert astar((0, 0), (11, 11), nb).expanded < bfs((0, 0), (11, 11), nb).expanded


def test_manhattan_is_admissible_on_layout():
    store = load_layout("Recorrido")
    goal = store.exit
    for cell in store.walkable:
        assert manhattan(cell, goal) <= bfs(cell, goal, store.neighbors).length


# ---------- tienda ----------

@pytest.mark.parametrize("name", list(LAYOUTS))
def test_layouts_are_valid_and_have_every_section(name):
    store = load_layout(name)
    assert store.validate() == []
    assert set(store.sections) == set(SECTIONS)
    assert (store.n_rows, store.n_cols) == (10, 16)


def test_invalid_layout_is_detected():
    store = Store("roto", ("#####",
                           "#E#F#",
                           "#.#C#",
                           "#S###"))
    problems = store.validate()
    assert any("Ropa mujer" in p for p in problems) and any("caja" in p for p in problems)


def test_section_mapping_from_dataset():
    assert section_for("Clothing", "Female") == "F"
    assert section_for("Clothing", "Male") == "M"
    assert section_for("Beauty", "Male") == section_for("Beauty", "Female") == "B"
    assert section_for("Electronics", "Female") == "T"


# ---------- simulación ----------

@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_simulation_finishes_and_counts_traffic(algorithm):
    sim = Simulation(load_layout("Pasillos"), algorithm, n_customers=25, seed=1, profiles=PROFILES).run_to_end()
    st = sim.stats()
    assert sim.finished and st["served"] == 25 and st["no_route"] == 0
    assert st["searches"] == 25 * 3                          # entrada->sección, sección->caja, caja->salida
    assert sim.visits.sum() == sum(len(c.path) for c in sim.customers)  # una visita por celda pisada
    assert sim.visits[sim.store.entrance] == 25 and sim.visits[sim.store.exit] == 25
    walls = np.array([[ch == "#" for ch in row] for row in sim.store.rows_text])
    assert sim.visits[walls].sum() == 0
    assert all(c.left is not None and c.left > c.arrival for c in sim.customers)


def test_simulation_is_deterministic_and_fps_independent():
    store = load_layout("Boutique")
    a = Simulation(store, "A*", 30, seed=7, profiles=PROFILES).run_to_end(dt=0.05)
    b = Simulation(store, "A*", 30, seed=7, profiles=PROFILES).run_to_end(dt=0.05)
    c = Simulation(store, "A*", 30, seed=7, profiles=PROFILES).run_to_end(dt=1 / 13)
    np.testing.assert_array_equal(a.visits, b.visits)
    np.testing.assert_array_equal(a.visits, c.visits)          # mismo recorrido con otro paso de tiempo
    assert abs(a.stats()["time"] - c.stats()["time"]) < 0.2


def test_same_customers_with_both_algorithms():
    store = load_layout("Pasillos")
    sims = [Simulation(store, algo, 40, seed=3, profiles=PROFILES).run_to_end() for algo in ALGORITHMS]
    assert sims[0].stats()["mean_route"] == sims[1].stats()["mean_route"]   # rutas óptimas de igual largo
    assert sims[1].stats()["mean_expanded"] < sims[0].stats()["mean_expanded"]


def test_hotspots_and_high_traffic_mask():
    sim = Simulation(load_layout("Recorrido"), "A*", 30, seed=0, profiles=PROFILES).run_to_end()
    top = sim.hotspots(5)
    assert [v for _, v in top] == sorted((v for _, v in top), reverse=True)
    mask = sim.high_traffic_mask()
    assert mask[top[0][0]] and mask.sum() < len(sim.store.walkable) / 2
    assert all(sim.store.kind(c) == "." for c, _ in sim.hotspots(3, aisles_only=True))


def test_profiles_follow_real_sales():
    pd = pytest.importorskip("pandas")
    from task_a_pathfinding.simulation import profiles_from_sales
    df = pd.DataFrame({"Product Category": ["Clothing", "Beauty"], "Gender": ["Male", "Female"],
                       "Quantity": [3, 1]})
    assert profiles_from_sales(df) == [Profile("M", 3), Profile("B", 1)]


# ---------- pantalla (sin ventana: driver de video dummy) ----------

def test_screen_opens_switches_and_returns_with_esc(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from common import ui
    from task_a_pathfinding import screen as screen_a

    screen = ui.create_window((1024, 640), "UrbanStyle")
    view = screen_a.PathfindingScreen(size=screen.get_size(), profiles=PROFILES)
    last_r, last_c = view.store.n_rows - 1, view.store.n_cols - 1
    drawn = view.cell_rect(1, 1).union(view.cell_rect(last_r, last_c - 1))   # todo lo que se dibuja de la tienda
    assert screen.get_rect().contains(drawn) and drawn.right < view.algo_buttons[0].rect.left
    click = lambda b: view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=b.rect.center))
    click(view.algo_buttons[0])
    assert view.algorithm == "BFS" and view.sim.algorithm == "BFS"
    click(view.layout_buttons[2])
    assert view.store.name == "Recorrido"
    for b in view.layer_buttons:
        click(b)                                   # activa y desactiva capas
    view.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=view.cell_center(8, 3), rel=(0, 0),
                                         buttons=(0, 0, 0)))
    for _ in range(30):
        view.update(0.1)
    view.layers.update({"Calor": True, "Rutas": True, "Búsqueda": True})
    view.draw(screen)                              # con todas las capas y el tooltip
    assert view.sim.time > 0

    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="", scancode=0))
    screen_a.run(screen, pygame.time.Clock())
    pygame.quit()
