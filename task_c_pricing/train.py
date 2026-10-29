"""Tarea C (Sebastián): entrena los agentes de precios con stable-baselines3 y los guarda en models/.

Uso: python -m task_c_pricing.train                 # los 3 agentes (~6 min en CPU)
     python -m task_c_pricing.train --only ppo      # solo uno: dqn, ppo o ppo_multi
Salidas:
  models/pricing_config.json                          parámetros de la demanda (los usan evaluate y la pantalla)
  models/dqn_pricing.zip, ppo_pricing.zip             un producto (Ropa): DQN y PPO (bonus: varios algoritmos)
  models/ppo_pricing_multi.zip                        tres productos (bonus: multiproducto; DQN no admite
                                                      acciones MultiDiscrete, por eso solo PPO)
  outputs/task_c/training/<agente>.monitor.csv        recompensa de cada episodio de entrenamiento
"""
import argparse
import time

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from common import config
from task_c_pricing.demand import CONFIG_PATH, PRODUCTS, build_config
from task_c_pricing.env import PricingEnv
from task_c_pricing.policies import model_path

TRAINING_DIR = config.OUTPUTS_DIR / "task_c" / "training"

# gamma = 0.9: el ingreso de hoy pesa más, pero el agente mira ~10 días adelante (cuánto tarda en mover el
# precio al óptimo de otro segmento). Red pequeña: el estado tiene 7 (o 15) números.
AGENTS = {
    "dqn": dict(algo=DQN, products=("Clothing",), timesteps=200_000, kwargs=dict(
        learning_rate=5e-4, buffer_size=50_000, learning_starts=2_000, batch_size=64, gamma=0.9,
        train_freq=4, target_update_interval=1_000, exploration_fraction=0.3, exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[64, 64]))),
    "ppo": dict(algo=PPO, products=("Clothing",), timesteps=200_000, kwargs=dict(
        learning_rate=3e-4, n_steps=1_200, batch_size=120, n_epochs=10, gamma=0.9, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]))),
    "ppo_multi": dict(algo=PPO, products=PRODUCTS, timesteps=400_000, kwargs=dict(
        learning_rate=3e-4, n_steps=1_200, batch_size=120, n_epochs=10, gamma=0.9, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]))),
}


def train(name: str, cfg, seed: int = config.SEED) -> None:
    spec = AGENTS[name]
    env = PricingEnv(cfg, spec["products"])
    check_env(env)  # compatibilidad con SB3 (no garantiza que el agente aprenda bien)
    env = Monitor(env, str(TRAINING_DIR / name))      # -> <name>.monitor.csv
    config.set_seed(seed)
    model = spec["algo"]("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **spec["kwargs"])
    t0 = time.time()
    model.learn(total_timesteps=spec["timesteps"])
    model.save(model_path(name))
    print(f"{name}: {spec['timesteps']:,} pasos en {time.time() - t0:.0f} s -> "
          f"{model_path(name).relative_to(config.ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(AGENTS), help="entrenar un solo agente")
    args = parser.parse_args()
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()
    cfg.save(CONFIG_PATH)
    print(f"Demanda guardada en {CONFIG_PATH.relative_to(config.ROOT)}")
    for name in [args.only] if args.only else list(AGENTS):
        train(name, cfg)


if __name__ == "__main__":
    main()
