"""Tarea C (Sebastián): RL vs. el MEJOR precio fijo (y DQN vs. PPO), en los mismos escenarios y semillas.

Uso: python -m task_c_pricing.evaluate        (~1 min; necesita los agentes de task_c_pricing.train)
Comparación justa:
  * los mismos EVAL_SEEDS para todas las políticas, y la demanda usa números aleatorios comunes: cada
    política enfrenta exactamente los mismos clientes con la misma disposición a pagar;
  * semillas de evaluación distintas de las de entrenamiento;
  * referencia = el MEJOR precio fijo (se prueban los 11 niveles), no uno arbitrario;
  * RL empieza en el mismo precio que el mejor precio fijo; aparte se mide qué pasa si empieza lejos;
  * se reportan media, desviación, IC 95 % (bootstrap) de la diferencia pareada y % de episodios ganados.
Políticas: precio fijo, DQN, PPO, oráculo (conoce la demanda real y va al óptimo del segmento con los mismos
movimientos permitidos: es un techo, no una política alcanzable) y precio al azar (control).
Salidas en outputs/task_c/ (ver outputs/task_c/README.md).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import config
from task_c_pricing.demand import PRODUCT_NAMES, PRODUCTS, SEGMENT_NAMES, load_config
from task_c_pricing.env import REWARD_SCALE, PricingEnv
from task_c_pricing.policies import (
    SUMMARY_JSON,
    agent_policy,
    keep_policy,
    load_agent,
    oracle_policy,
    random_policy_factory,
    run_episode,
)
from task_c_pricing.train import AGENTS, TRAINING_DIR

OUT_DIR = config.OUTPUTS_DIR / "task_c"
METRICS_DIR, FIGURES_DIR = OUT_DIR / "metrics", OUT_DIR / "figures"
EVAL_SEEDS = range(10_000, 10_200)        # 200 episodios de 60 días
N_BOOT = 2_000
COLORS = {"PPO": "#FF5A36", "DQN": "#C9644A", "Oráculo": "#7E9C89", "Mejor precio fijo": "#B09A74",
          "Precio al azar": "#77716B"}


# ---------- episodios ----------

def evaluate_policy(env, policy, initial_level, seeds=EVAL_SEEDS) -> np.ndarray:
    return np.array([run_episode(env, policy, s, initial_level)["revenue"] for s in seeds])


def fixed_price_sweep(env, seeds=EVAL_SEEDS) -> pd.DataFrame:
    """Ingreso por producto con cada nivel de precio fijo (todos los productos en el mismo nivel; como la
    demanda de cada producto es independiente, el mejor precio fijo de cada uno se elige por separado)."""
    rows = []
    for level in range(env.cfg.n_levels):
        per_product = np.array([run_episode(env, keep_policy, s, level)["revenue_by_product"] for s in seeds])
        for p, product in enumerate(env.cfg.products):
            rows.append({"product": product, "level": level, "multiplier": env.cfg.price_multipliers[level],
                         "price": round(env.cfg.price(p, level), 2), "mean_revenue": per_product[:, p].mean(),
                         "std_revenue": per_product[:, p].std(ddof=1)})
    return pd.DataFrame(rows)


def paired_summary(results: dict, reference: str, rng=None) -> pd.DataFrame:
    """Media, desviación e IC 95 % bootstrap de la diferencia pareada contra la referencia."""
    rng = rng or np.random.default_rng(config.SEED)
    base = results[reference]
    rows = []
    for name, rev in results.items():
        diff = rev - base
        boots = np.array([rng.choice(diff, len(diff)).mean() for _ in range(N_BOOT)])
        rows.append({"policy": name, "mean_revenue": rev.mean(), "std_revenue": rev.std(ddof=1),
                     "diff_vs_fixed": diff.mean(), "diff_ci_lo": np.quantile(boots, 0.025),
                     "diff_ci_hi": np.quantile(boots, 0.975), "gain_pct": 100 * diff.mean() / base.mean(),
                     "win_rate_pct": 100 * (diff > 0).mean()})
    return pd.DataFrame(rows)


def policy_by_segment(env, policies: dict, seeds=EVAL_SEEDS, initial_level=5) -> pd.DataFrame:
    """Precio medio que cobra cada política según el segmento del día (¿aprendió a diferenciar?)."""
    rows = []
    for name, policy in policies.items():
        eps = [run_episode(env, policy, s, initial_level) for s in seeds[:50]]
        prices, segs = np.concatenate([e["prices"][:, 0] for e in eps]), np.concatenate([e["segments"] for e in eps])
        for s, seg in enumerate(env.cfg.segments):
            rows.append({"policy": name, "segment": SEGMENT_NAMES[seg],
                         "mean_multiplier": prices[segs == s].mean() / env.cfg.ref_prices[0]})
    for s, seg in enumerate(env.cfg.segments):
        rows.append({"policy": "Óptimo del segmento", "segment": SEGMENT_NAMES[seg],
                     "mean_multiplier": env.cfg.price_multipliers[env.cfg.optimal_level(s, 0)]})
    return pd.DataFrame(rows).pivot(index="segment", columns="policy", values="mean_multiplier")


# ---------- figuras ----------

def plot_fixed_curve(sweep: pd.DataFrame, summary: pd.DataFrame, product="Clothing") -> None:
    d = sweep[sweep["product"] == product]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(d["price"], d["mean_revenue"] / 1e3, "o-", color=COLORS["Mejor precio fijo"], label="Precio fijo")
    for name, style in (("PPO", "--"), ("DQN", ":"), ("Oráculo", "-.")):
        row = summary[summary["policy"] == name]
        if len(row):
            ax.axhline(row["mean_revenue"].iloc[0] / 1e3, color=COLORS[name], ls=style, lw=1.5, label=name)
    best = d.loc[d["mean_revenue"].idxmax()]
    ax.annotate(f"mejor fijo: ${best['price']:.0f}", (best["price"], best["mean_revenue"] / 1e3),
                textcoords="offset points", xytext=(0, -24), ha="center")
    ax.set_xlabel(f"precio fijo de {PRODUCT_NAMES[product]} ($)")
    ax.set_ylabel("ingreso por episodio (miles $)")
    ax.set_title("Ningún precio fijo alcanza al precio dinámico (200 episodios de 60 días)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES_DIR / "fixed_price_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(summary: pd.DataFrame, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    s = summary.sort_values("mean_revenue")
    colors = [COLORS.get(p, "#77716B") for p in s["policy"]]
    ax.barh(s["policy"], s["mean_revenue"] / 1e3, xerr=s["std_revenue"] / 1e3, color=colors, capsize=3)
    for y, (_, row) in enumerate(s.iterrows()):
        ax.text(row["mean_revenue"] / 1e3 * 0.02, y, f"{row['gain_pct']:+.1f} %", va="center", color="white",
                fontsize=8)
    ax.set_xlabel("ingreso por episodio (miles $, media ± desviación)")
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_episode(env, policies: dict, seed: int, initial_level: int) -> None:
    """Un mismo episodio: precio de cada política según el segmento del día e ingreso acumulado."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    shades = ["#F3E3D3", "#D9E4DC", "#E4DCEF"]
    eps = {name: run_episode(env, pol, seed, initial_level) for name, pol in policies.items()}
    segs = next(iter(eps.values()))["segments"]
    days = np.arange(1, len(segs) + 1)
    for d, s in zip(days, segs):
        ax1.axvspan(d - 0.5, d + 0.5, color=shades[s], lw=0)
    for s, seg in enumerate(env.cfg.segments):
        ax1.plot([], [], color=shades[s], lw=8, label=SEGMENT_NAMES[seg])
    for name, ep in eps.items():
        ax1.step(days, ep["prices"][:, 0], where="mid", color=COLORS[name], lw=2, label=name)
        ax2.plot(days, np.cumsum(ep["daily_revenue"]) / 1e3, color=COLORS[name], lw=2, label=name)
    ax1.set_ylabel("precio ($)")
    ax1.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    ax2.set_ylabel("ingreso acumulado (miles $)")
    ax2.set_xlabel("día")
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle("Un episodio de evaluación: el precio sigue al segmento del día")
    fig.savefig(FIGURES_DIR / "episode_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(references: dict) -> None:
    fig, axes = plt.subplots(1, len(AGENTS), figsize=(5 * len(AGENTS), 3.4))
    for ax, name in zip(axes, AGENTS):
        path = TRAINING_DIR / f"{name}.monitor.csv"
        if not path.exists():
            continue
        log = pd.read_csv(path, skiprows=1)
        steps = log["l"].cumsum()
        revenue = log["r"] * REWARD_SCALE / 1e3
        ax.plot(steps, revenue, color="#D8D2CA", lw=0.5)
        ax.plot(steps, revenue.rolling(50, min_periods=1).mean(), color=COLORS["PPO"], lw=2, label="media móvil 50 ep.")
        for label, value in references.get(name, {}).items():
            ax.axhline(value / 1e3, ls="--", lw=1.2, color=COLORS[label], label=label)
        ax.set_title(name.upper().replace("_MULTI", " multiproducto"))
        ax.set_xlabel("pasos de entrenamiento")
        ax.set_ylabel("ingreso por episodio (miles $)")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Curvas de aprendizaje (precio inicial al azar en cada episodio)")
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- principal ----------

def evaluate_single(cfg, agents: dict):
    env = PricingEnv(cfg, ("Clothing",))
    sweep = fixed_price_sweep(env)
    best_level = int(sweep.loc[sweep["mean_revenue"].idxmax(), "level"])
    policies = {"Mejor precio fijo": keep_policy, **{k: agent_policy(m) for k, m in agents.items()},
                "Oráculo": oracle_policy, "Precio al azar": random_policy_factory()}
    results = {name: evaluate_policy(env, pol, best_level) for name, pol in policies.items()}
    summary = paired_summary(results, "Mejor precio fijo")

    # ¿Y si el agente arranca lejos del mejor precio? (lo que permite el slider de la pantalla)
    start_rows = []
    for level in (0, 3, 5, 8, 10):
        row = {"initial_multiplier": cfg.price_multipliers[level],
               "fixed_at_initial": evaluate_policy(env, keep_policy, level).mean(),
               "best_fixed": results["Mejor precio fijo"].mean()}
        for name, m in agents.items():
            row[name] = evaluate_policy(env, agent_policy(m), level).mean()
        start_rows.append(row)
    by_start = pd.DataFrame(start_rows)
    by_segment = policy_by_segment(env, {k: agent_policy(m) for k, m in agents.items()} | {"Oráculo": oracle_policy})
    episodes = pd.DataFrame({"seed": list(EVAL_SEEDS), **results})
    plot_episode(env, {"Mejor precio fijo": keep_policy, **{"PPO": policies["PPO"]}, "Oráculo": oracle_policy},
                 EVAL_SEEDS[0], best_level)
    return sweep, best_level, summary, by_start, by_segment, episodes


def evaluate_multi(cfg, model):
    env = PricingEnv(cfg, PRODUCTS)
    sweep = fixed_price_sweep(env)
    best_levels = [int(sweep[sweep["product"] == p].set_index("level")["mean_revenue"].idxmax()) for p in PRODUCTS]
    policies = {"Mejor precio fijo": keep_policy, "PPO": agent_policy(model), "Oráculo": oracle_policy,
                "Precio al azar": random_policy_factory()}
    results = {name: evaluate_policy(env, pol, best_levels) for name, pol in policies.items()}
    return sweep, best_levels, paired_summary(results, "Mejor precio fijo")


def main() -> None:
    for folder in (METRICS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    agents = {"DQN": load_agent("dqn"), "PPO": load_agent("ppo")}

    sweep, best_level, summary, by_start, by_segment, episodes = evaluate_single(cfg, agents)
    sweep.to_csv(METRICS_DIR / "fixed_prices_single.csv", index=False, float_format="%.2f")
    summary.to_csv(METRICS_DIR / "comparison_single.csv", index=False, float_format="%.2f")
    by_start.to_csv(METRICS_DIR / "initial_price_single.csv", index=False, float_format="%.2f")
    by_segment.to_csv(METRICS_DIR / "policy_by_segment.csv", float_format="%.2f")
    episodes.to_csv(METRICS_DIR / "episodes_single.csv", index=False, float_format="%.2f")

    msweep, best_levels, msummary = evaluate_multi(cfg, load_agent("ppo_multi"))
    msweep.to_csv(METRICS_DIR / "fixed_prices_multi.csv", index=False, float_format="%.2f")
    msummary.to_csv(METRICS_DIR / "comparison_multi.csv", index=False, float_format="%.2f")

    plot_fixed_curve(sweep, summary)
    plot_comparison(summary, "Un producto (Ropa): 200 episodios, mismas semillas",
                    FIGURES_DIR / "comparison_single.png")
    plot_comparison(msummary, "Tres productos: 200 episodios, mismas semillas", FIGURES_DIR / "comparison_multi.png")
    ref = lambda s, name: float(s.loc[s["policy"] == name, "mean_revenue"].iloc[0])
    plot_learning_curves({
        "dqn": {"Mejor precio fijo": ref(summary, "Mejor precio fijo"), "Oráculo": ref(summary, "Oráculo")},
        "ppo": {"Mejor precio fijo": ref(summary, "Mejor precio fijo"), "Oráculo": ref(summary, "Oráculo")},
        "ppo_multi": {"Mejor precio fijo": ref(msummary, "Mejor precio fijo"), "Oráculo": ref(msummary, "Oráculo")},
    })

    summary_json = {
        "eval_episodes": len(EVAL_SEEDS),
        "single": {"product": "Clothing", "best_fixed_level": best_level,
                   "best_fixed_multiplier": cfg.price_multipliers[best_level],
                   **{row["policy"]: {k: round(float(row[k]), 2) for k in
                                      ("mean_revenue", "gain_pct", "diff_ci_lo", "diff_ci_hi", "win_rate_pct")}
                      for _, row in summary.iterrows()}},
        "multi": {"products": list(PRODUCTS), "best_fixed_levels": best_levels,
                  **{row["policy"]: {k: round(float(row[k]), 2) for k in
                                     ("mean_revenue", "gain_pct", "diff_ci_lo", "diff_ci_hi", "win_rate_pct")}
                     for _, row in msummary.iterrows()}},
    }
    SUMMARY_JSON.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 160)
    print("Un producto (Ropa), referencia = mejor precio fijo "
          f"({cfg.price_multipliers[best_level]:.0%} del precio de referencia):")
    print(summary.round(1).to_string(index=False))
    print("\nPrecio medio cobrado por segmento (× precio de referencia):")
    print(by_segment.round(2).to_string())
    print("\nSi el agente arranca en otro precio:")
    print(by_start.round(0).assign(initial_multiplier=by_start["initial_multiplier"]).to_string(index=False))
    print(f"\nTres productos (mejores niveles fijos {best_levels}):")
    print(msummary.round(1).to_string(index=False))
    print(f"\nSalidas en {OUT_DIR.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
