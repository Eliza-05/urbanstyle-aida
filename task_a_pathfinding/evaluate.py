"""Tarea A (Sebastián): comparación BFS vs. A* y experimento de distribuciones (layouts) de la tienda.

Uso: python -m task_a_pathfinding.evaluate
Salidas en outputs/task_a/ (ver outputs/task_a/README.md):
  metrics/algorithm_pairs.csv     BFS y A* en los mismos pares origen-destino de cada layout
  metrics/algorithm_summary.csv   resumen por layout: largo, nodos expandidos y tiempo
  metrics/layouts.csv             tráfico por layout (promedio de varias semillas)
  traffic/traffic_<layout>.csv    visitas, ocupación y pico por celda
  figures/heatmaps.png            mapa de calor de cada layout
  figures/exploration.png         celdas que expande BFS y A* en un mismo trayecto
  figures/expanded.png            nodos expandidos por layout y algoritmo
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import config
from task_a_pathfinding.search import ALGORITHMS
from task_a_pathfinding.simulation import Simulation, load_profiles
from task_a_pathfinding.store import CELL_NAMES, LAYOUTS, SECTIONS, WALL, Store, load_layout

OUT_DIR = config.OUTPUTS_DIR / "task_a"
METRICS_DIR, TRAFFIC_DIR, FIGURES_DIR = OUT_DIR / "metrics", OUT_DIR / "traffic", OUT_DIR / "figures"

N_CUSTOMERS = 100      # clientes por simulación
SEEDS = range(10)      # semillas del experimento de layouts
TIMING_REPEATS = 25    # repeticiones por par para medir el tiempo (se reporta la mediana)


def key_points(store: Store) -> list:
    """Orígenes y destinos reales de un cliente: entrada, celdas de sección, cajas y salida."""
    sections = [c for cells in store.sections.values() for c in cells]
    return [store.entrance, *sections, *store.checkouts, store.exit]


def compare_algorithms(store: Store, repeats: int = TIMING_REPEATS) -> pd.DataFrame:
    """BFS y A* en TODOS los pares (origen, destino) de puntos clave: mismas condiciones para los dos."""
    rows = []
    points = key_points(store)
    for start in points:
        for goal in points:
            if start == goal:
                continue
            row = {"layout": store.name, "start": start, "goal": goal}
            for name, algo in ALGORITHMS.items():
                result = algo(start, goal, store.neighbors)
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    algo(start, goal, store.neighbors)
                    times.append(time.perf_counter() - t0)
                row[f"{name}_length"] = result.length
                row[f"{name}_expanded"] = result.expanded
                row[f"{name}_us"] = float(np.median(times) * 1e6)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_algorithms(pairs: pd.DataFrame) -> pd.DataFrame:
    g = pairs.groupby("layout", sort=False)
    out = pd.DataFrame({
        "pairs": g.size(),
        "same_length_pct": g.apply(lambda d: 100 * (d["BFS_length"] == d["A*_length"]).mean(),
                                   include_groups=False),
        "mean_length": g["A*_length"].mean(),
        "bfs_expanded": g["BFS_expanded"].mean(),
        "astar_expanded": g["A*_expanded"].mean(),
        "astar_fewer_pct": g.apply(lambda d: 100 * (d["A*_expanded"] < d["BFS_expanded"]).mean(),
                                   include_groups=False),
        "astar_equal_pct": g.apply(lambda d: 100 * (d["A*_expanded"] == d["BFS_expanded"]).mean(),
                                   include_groups=False),
        "bfs_us": g["BFS_us"].mean(),
        "astar_us": g["A*_us"].mean(),
    })
    out["expanded_ratio"] = out["astar_expanded"] / out["bfs_expanded"]
    return out.reset_index()


def traffic_table(sim: Simulation) -> pd.DataFrame:
    store = sim.store
    return pd.DataFrame([{"row": r, "col": c, "kind": CELL_NAMES[store.kind((r, c))],
                          "description": store.describe((r, c)), "visits": int(sim.visits[r, c]),
                          "occupancy_s": round(float(sim.occupancy[r, c]), 2), "peak": int(sim.peak[r, c])}
                         for r, c in store.walkable])


def layout_experiment(profiles, algorithm: str = "A*") -> tuple[pd.DataFrame, dict]:
    """Mismos clientes (semillas) en los 3 layouts. Devuelve la tabla y las visitas promedio por layout."""
    rows, mean_visits = [], {}
    for name in LAYOUTS:
        store = load_layout(name)
        visits = []
        for seed in SEEDS:
            sim = Simulation(store, algorithm, N_CUSTOMERS, seed=seed, profiles=profiles).run_to_end()
            st = sim.stats()
            top5 = sum(v for _, v in sim.hotspots(5))
            aisle_cell, aisle_visits = sim.hotspots(1, aisles_only=True)[0]
            rows.append({"layout": name, "seed": seed, "mean_route": st["mean_route"],
                         "time_in_store": st["mean_time_in_store"],
                         "total_time": st["time"], "max_peak": st["max_peak"],
                         "top5_share_pct": 100 * top5 / sim.visits.sum(),
                         "hottest_aisle": store.describe(aisle_cell), "hottest_aisle_visits": aisle_visits,
                         "mean_expanded": st["mean_expanded"]})
            visits.append(sim.visits)
            if seed == SEEDS[0]:
                traffic_table(sim).to_csv(TRAFFIC_DIR / f"traffic_{name.lower()}.csv", index=False)
        mean_visits[name] = np.mean(visits, axis=0)
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("layout", sort=False).agg(
        mean_route=("mean_route", "mean"), mean_route_std=("mean_route", "std"),
        time_in_store=("time_in_store", "mean"),
        total_time=("total_time", "mean"), max_peak=("max_peak", "mean"),
        top5_share_pct=("top5_share_pct", "mean"), mean_expanded=("mean_expanded", "mean"),
        hottest_aisle=("hottest_aisle", lambda s: s.mode().iloc[0]),
        hottest_aisle_visits=("hottest_aisle_visits", "mean"),
    ).reset_index()
    return summary, mean_visits


def _store_background(ax, store: Store) -> None:
    walls = np.array([[1.0 if ch == WALL else 0.0 for ch in row] for row in store.rows_text])
    ax.imshow(np.ma.masked_equal(walls, 0), cmap="Greys", vmin=0, vmax=1.4)
    for r, row in enumerate(store.rows_text):
        for c, ch in enumerate(row):
            if ch in SECTIONS or ch in "ECS":
                ax.text(c, r, ch, ha="center", va="center", fontsize=8, color="#333")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_heatmaps(mean_visits: dict) -> None:
    fig, axes = plt.subplots(1, len(mean_visits), figsize=(5 * len(mean_visits), 3.6))
    vmax = max(v.max() for v in mean_visits.values())
    for ax, (name, visits) in zip(axes, mean_visits.items()):
        store = load_layout(name)
        im = ax.imshow(np.ma.masked_equal(visits, 0), cmap="OrRd", vmin=0, vmax=vmax)
        _store_background(ax, store)
        ax.set_title(f"{name}")
    fig.colorbar(im, ax=axes, shrink=0.8, label=f"visitas por celda (media, {N_CUSTOMERS} clientes)")
    fig.suptitle("Tráfico por celda en cada distribución (A*, 10 semillas)")
    fig.savefig(FIGURES_DIR / "heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_exploration(layout: str = "Pasillos") -> None:
    """Mismo trayecto (entrada -> Ropa hombre): celdas que expande cada algoritmo y la ruta encontrada."""
    store = load_layout(layout)
    start, goal = store.entrance, store.sections["M"][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (name, algo) in zip(axes, ALGORITHMS.items()):
        result = algo(start, goal, store.neighbors)
        order = np.full((store.n_rows, store.n_cols), np.nan)
        for i, cell in enumerate(result.explored):
            order[cell] = i
        ax.imshow(order, cmap="Blues_r", vmin=0, vmax=len(result.explored) * 1.3)
        _store_background(ax, store)
        ys, xs = zip(*result.path)
        ax.plot(xs, ys, color="#FF5A36", lw=2.5)
        ax.set_title(f"{name}: {result.expanded} nodos expandidos, ruta de {result.length} pasos")
    fig.suptitle(f"Exploración en '{layout}': entrada -> Ropa hombre (más oscuro = expandido antes)")
    fig.savefig(FIGURES_DIR / "exploration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_expanded(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["bfs_expanded"], 0.36, label="BFS", color="#77716B")
    ax.bar(x + 0.18, summary["astar_expanded"], 0.36, label="A*", color="#FF5A36")
    ax.set_xticks(x, summary["layout"])
    ax.set_ylabel("nodos expandidos (media por búsqueda)")
    ax.set_title("Mismas rutas óptimas, distinto esfuerzo de búsqueda")
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "expanded.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for folder in (METRICS_DIR, TRAFFIC_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    for name in LAYOUTS:
        problems = load_layout(name).validate()
        if problems:
            raise ValueError(f"Layout {name!r} inválido: {problems}")

    pairs = pd.concat([compare_algorithms(load_layout(name)) for name in LAYOUTS], ignore_index=True)
    pairs.to_csv(METRICS_DIR / "algorithm_pairs.csv", index=False)
    summary = summarize_algorithms(pairs)
    summary.to_csv(METRICS_DIR / "algorithm_summary.csv", index=False, float_format="%.3f")
    if (summary["same_length_pct"] < 100).any():
        raise AssertionError("BFS y A* dieron rutas de distinto largo: revisar la heurística")

    layouts, mean_visits = layout_experiment(load_profiles())
    layouts.to_csv(METRICS_DIR / "layouts.csv", index=False, float_format="%.3f")
    plot_heatmaps(mean_visits)
    plot_exploration()
    plot_expanded(summary)

    pd.set_option("display.width", 160)
    print("BFS vs. A* (todos los pares de puntos clave):")
    print(summary.round(2).to_string(index=False))
    print("\nExperimento de layouts (A*, media de", len(SEEDS), "semillas):")
    print(layouts.round(2).to_string(index=False))
    print(f"\nSalidas en {OUT_DIR.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
