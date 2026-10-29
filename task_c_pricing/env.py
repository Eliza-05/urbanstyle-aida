"""Tarea C (Sebastián): entorno Gymnasium de fijación dinámica de precios (uno o varios productos).

Un paso = un día de ventas; un episodio = EPISODE_DAYS días.

  Estado (observación, todo en [0, 1] o cerca):
    * precio actual de cada producto: nivel / (niveles - 1);
    * segmento que predomina HOY (one-hot de 3);
    * ventas recientes: unidades de los últimos HISTORY días de cada producto / unidades de referencia.
  Acción: ajustar el precio de cada producto en la grilla (50 % … 150 % del precio de referencia, pasos de
    10 %): MOVES = bajar 2 niveles, bajar 1, mantener, subir 1, subir 2. Discreta para que sirvan DQN y PPO:
    Discrete(5) con un producto; MultiDiscrete([5] * n) con varios (solo PPO, DQN no admite MultiDiscrete).
  Recompensa: ingreso del día (Σ precio × unidades) / REWARD_SCALE. El ingreso sin escalar va en info.

Orden de un paso: el agente ve el segmento de hoy -> fija el precio -> llegan los clientes de hoy y compran
-> mañana puede predominar otro segmento (cadena de Markov).
El precio siempre queda dentro de la grilla (se recorta en los extremos).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from task_c_pricing.demand import DemandConfig, load_config

MOVES = (-2, -1, 0, 1, 2)
MOVE_NAMES = ("Bajar 20 %", "Bajar 10 %", "Mantener", "Subir 10 %", "Subir 20 %")
KEEP = MOVES.index(0)
EPISODE_DAYS = 60
HISTORY = 3
REWARD_SCALE = 1000.0          # recompensa en miles de $ (misma decisión óptima, números cómodos para la red)
SALES_CLIP = 5.0


class PricingEnv(gym.Env):
    """Precios dinámicos para UrbanStyle. reset(options={"initial_level": k} o {"initial_multiplier": 0.8})."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: DemandConfig | None = None, products: tuple[str, ...] = ("Clothing",),
                 episode_days: int = EPISODE_DAYS, history: int = HISTORY):
        super().__init__()
        self.cfg = (cfg or load_config()).subset(tuple(products))
        self.n_products = len(self.cfg.products)
        self.n_segments = len(self.cfg.segments)
        self.episode_days, self.history = episode_days, history
        if self.n_products == 1:
            self.action_space = spaces.Discrete(len(MOVES))
        else:
            self.action_space = spaces.MultiDiscrete([len(MOVES)] * self.n_products)
        n_obs = self.n_products + self.n_segments + self.n_products * history
        high = np.concatenate([np.ones(self.n_products + self.n_segments), np.full(self.n_products * history,
                                                                                    SALES_CLIP)])
        self.observation_space = spaces.Box(0.0, high.astype(np.float32), shape=(n_obs,), dtype=np.float32)
        self.ref_units = np.array([self.cfg.reference_units(p) for p in range(self.n_products)])
        self.levels = np.zeros(self.n_products, dtype=int)
        self.segment, self.day = 0, 0
        self.sales = np.zeros((history, self.n_products))

    # ---------- ayudas ----------
    def deltas(self, action) -> np.ndarray:
        """Acción -> cambio de nivel de cada producto."""
        action = np.atleast_1d(np.asarray(action, dtype=int))
        return np.array([MOVES[a] for a in action])

    def prices(self) -> np.ndarray:
        return np.array([self.cfg.price(p, k) for p, k in enumerate(self.levels)])

    def _obs(self) -> np.ndarray:
        segment = np.zeros(self.n_segments)
        segment[self.segment] = 1.0
        sales = np.clip(self.sales / self.ref_units, 0, SALES_CLIP).ravel()
        return np.concatenate([self.levels / (self.cfg.n_levels - 1), segment, sales]).astype(np.float32)

    # ---------- API Gymnasium ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "initial_level" in options:
            levels = np.broadcast_to(options["initial_level"], (self.n_products,))
        elif "initial_multiplier" in options:
            m = options["initial_multiplier"]
            levels = [self.cfg.level_of(p, m * self.cfg.ref_prices[p]) for p in range(self.n_products)]
        else:  # al entrenar: precio inicial al azar, para que el agente sepa salir de cualquier precio
            levels = self.np_random.integers(0, self.cfg.n_levels, self.n_products)
        self.levels = np.clip(np.array(levels, dtype=int), 0, self.cfg.n_levels - 1)
        self.segment = int(self.np_random.choice(self.n_segments, p=self.cfg.segment_share))
        self.day = 0
        # Historial inicial: lo que se vendería en promedio con el precio inicial (sin información extra).
        expected = [self.cfg.expected_units(self.cfg.price(p, k), self.segment, p) for p, k in enumerate(self.levels)]
        self.sales = np.tile(expected, (self.history, 1))
        return self._obs(), {"prices": self.prices(), "segment": self.segment}

    def step(self, action):
        self.levels = np.clip(self.levels + self.deltas(action), 0, self.cfg.n_levels - 1)
        prices = self.prices()
        today = self.segment
        day = self.cfg.sample_day(self.np_random, today)
        units = np.array([self.cfg.units_sold(wtp, qty, prices[p]) for p, (wtp, qty) in enumerate(day)])
        revenue = float(np.dot(prices, units))
        self.sales = np.vstack([self.sales[1:], units])
        self.segment = self.cfg.next_segment(self.np_random, today)
        self.day += 1
        info = {"revenue": revenue, "units": units, "prices": prices, "segment": today,
                "next_segment": self.segment, "shoppers": np.array([len(w) for w, _ in day])}
        return self._obs(), revenue / REWARD_SCALE, False, self.day >= self.episode_days, info


def make_env(products: tuple[str, ...] = ("Clothing",), cfg: DemandConfig | None = None) -> PricingEnv:
    return PricingEnv(cfg, products)
