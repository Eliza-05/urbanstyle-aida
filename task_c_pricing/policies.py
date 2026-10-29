"""Tarea C (Sebastián): políticas de precios que se comparan (las usan evaluate.py y la pantalla).

Una política es una función (obs, env) -> acción:
  * keep_policy:   precio fijo (siempre "mantener");
  * agent_policy:  agente de stable-baselines3 ya entrenado (determinista);
  * oracle_policy: conoce la demanda REAL y lleva cada precio al óptimo del segmento de hoy con los mismos
                   movimientos permitidos (±2 niveles). Es un techo de referencia, no una política alcanzable;
  * random_policy_factory: movimientos al azar (control).
"""
import json

import numpy as np

from common import config
from task_c_pricing.env import KEEP, MOVES, PricingEnv

SUMMARY_JSON = config.OUTPUTS_DIR / "task_c" / "metrics" / "summary.json"


def keep_policy(obs, env: PricingEnv):
    return KEEP if env.n_products == 1 else [KEEP] * env.n_products


def oracle_policy(obs, env: PricingEnv):
    moves = [MOVES.index(int(np.clip(env.cfg.optimal_level(env.segment, p) - env.levels[p], -2, 2)))
             for p in range(env.n_products)]
    return moves[0] if env.n_products == 1 else moves


def random_policy_factory(seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    return lambda obs, env: (rng.integers(len(MOVES)) if env.n_products == 1
                             else rng.integers(len(MOVES), size=env.n_products))


def agent_policy(model):
    return lambda obs, env: model.predict(obs, deterministic=True)[0]


def model_path(name: str):
    """dqn -> models/dqn_pricing.zip, ppo_multi -> models/ppo_pricing_multi.zip."""
    algo, _, multi = name.partition("_")
    return config.MODELS_DIR / f"{algo}_pricing{'_' + multi if multi else ''}.zip"


def load_agent(name: str):
    from stable_baselines3 import DQN, PPO  # import diferido: torch solo se carga cuando hace falta
    algo = DQN if name.startswith("dqn") else PPO
    return algo.load(model_path(name), device="cpu")


def best_fixed_levels(env: PricingEnv) -> list[int]:
    """Mejor nivel de precio fijo de cada producto: el medido por evaluate.py (summary.json) si existe; si no,
    el que maximiza el ingreso esperado con la participación de largo plazo de cada segmento."""
    if SUMMARY_JSON.exists():
        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
        if env.n_products == 1 and env.cfg.products == (summary["single"]["product"],):
            return [summary["single"]["best_fixed_level"]]
        if list(env.cfg.products) == summary["multi"]["products"]:
            return summary["multi"]["best_fixed_levels"]
    cfg = env.cfg
    return [int(np.argmax([sum(share * cfg.expected_revenue(cfg.price(p, k), s, p)
                               for s, share in enumerate(cfg.segment_share)) for k in range(cfg.n_levels)]))
            for p in range(env.n_products)]


def run_episode(env: PricingEnv, policy, seed: int, initial_level) -> dict:
    """Un episodio completo; devuelve el ingreso total y las trayectorias diarias."""
    obs, _ = env.reset(seed=seed, options={"initial_level": initial_level})
    days = []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs, env))
        days.append(info)
        done = terminated or truncated
    return {"revenue": sum(d["revenue"] for d in days),
            "revenue_by_product": np.sum([d["prices"] * d["units"] for d in days], axis=0),
            "prices": np.array([d["prices"] for d in days]),
            "segments": np.array([d["segment"] for d in days]),
            "daily_revenue": np.array([d["revenue"] for d in days])}
