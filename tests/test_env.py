"""Pruebas de la Tarea C: demanda, entorno Gymnasium, políticas y pantalla (no entrenan ni necesitan el CSV)."""
import numpy as np
import pygame
import pytest
from stable_baselines3.common.env_checker import check_env

from task_c_pricing.demand import PRICE_MULTIPLIERS, PRODUCTS, SEGMENTS, WTP_SCALE, DemandConfig
from task_c_pricing.env import EPISODE_DAYS, KEEP, MOVES, REWARD_SCALE, PricingEnv
from task_c_pricing.policies import best_fixed_levels, keep_policy, oracle_policy, run_episode


def toy_config() -> DemandConfig:
    """Escalas como las del dataset, sin leer el CSV."""
    return DemandConfig(
        products=PRODUCTS, ref_prices=(174.0, 184.0, 182.0), segments=SEGMENTS, segment_share=(0.37, 0.43, 0.20),
        qty_probs=((0.25, 0.25, 0.25, 0.25),) * 3,
        wtp_scale=tuple(tuple(WTP_SCALE[s][p] for p in PRODUCTS) for s in SEGMENTS))


CFG = toy_config()


# ---------- demanda ----------

@pytest.mark.parametrize("segment", range(3))
def test_demand_decreases_with_price(segment):
    prices = np.linspace(50, 400, 30)
    p_buy = CFG.buy_probability(prices, segment, 0)
    assert np.all(np.diff(p_buy) < 0) and p_buy[0] < 1 and p_buy[-1] > 0


def test_segments_have_different_optimal_prices():
    """Si todos tuvieran el mismo óptimo, un precio fijo bastaría y el RL no tendría nada que aprender."""
    for p in range(len(PRODUCTS)):
        optima = {CFG.optimal_level(s, p) for s in range(len(SEGMENTS))}
        assert len(optima) == 3
        assert all(0 < k < CFG.n_levels - 1 for k in optima)   # óptimos interiores: subir al máximo no conviene


def test_simulated_units_match_expected():
    rng = np.random.default_rng(0)
    price = CFG.price(0, 5)
    units = [CFG.units_sold(*CFG.sample_day(rng, 1)[0], price) for _ in range(3000)]
    assert np.mean(units) == pytest.approx(CFG.expected_units(price, 1, 0), rel=0.03)


def test_config_roundtrip_and_subset(tmp_path):
    path = tmp_path / "pricing_config.json"
    CFG.save(path)
    assert DemandConfig.load(path) == CFG
    one = CFG.subset(("Beauty",))
    assert one.ref_prices == (184.0,) and one.wtp_scale[0] == (WTP_SCALE["18-35"]["Beauty"],)


# ---------- entorno ----------

@pytest.mark.parametrize("products", [("Clothing",), PRODUCTS])
def test_env_passes_check_env(products):
    check_env(PricingEnv(CFG, products))


def test_observation_contains_price_segment_and_recent_sales():
    env = PricingEnv(CFG)
    obs, _ = env.reset(seed=1, options={"initial_level": 4})
    assert obs.shape == (1 + 3 + 3,)
    assert obs[0] == pytest.approx(4 / 10)                       # precio
    assert obs[1:4].sum() == 1 and obs[1 + env.segment] == 1     # segmento (one-hot)
    obs, *_ = env.step(KEEP)
    assert obs[-1] == pytest.approx(min(env.sales[-1, 0] / env.ref_units[0], 5), rel=1e-5)  # ventas recientes


def test_actions_move_price_and_stay_in_bounds():
    env = PricingEnv(CFG)
    env.reset(seed=0, options={"initial_level": 1})
    env.step(MOVES.index(-2))
    assert env.levels[0] == 0                                     # se recorta en el mínimo
    for _ in range(10):
        env.step(MOVES.index(2))
    assert env.levels[0] == CFG.n_levels - 1                      # y en el máximo
    _, _, _, _, info = env.step(MOVES.index(-1))
    assert info["prices"][0] == pytest.approx(CFG.ref_prices[0] * PRICE_MULTIPLIERS[-2])


def test_reward_is_revenue_and_episode_is_truncated():
    env = PricingEnv(CFG)
    env.reset(seed=3, options={"initial_multiplier": 1.0})
    steps = 0
    while True:
        _, reward, terminated, truncated, info = env.step(KEEP)
        steps += 1
        assert reward == pytest.approx(info["revenue"] / REWARD_SCALE)
        assert info["revenue"] == pytest.approx(float(np.dot(info["prices"], info["units"])))
        assert not terminated
        if truncated:
            break
    assert steps == EPISODE_DAYS


def test_same_seed_same_customers_for_any_policy():
    """Números aleatorios comunes: los segmentos y clientes no dependen de lo que haga la política."""
    env = PricingEnv(CFG)
    a = run_episode(env, keep_policy, 11, 5)
    b = run_episode(env, oracle_policy, 11, 5)
    c = run_episode(env, keep_policy, 11, 5)
    np.testing.assert_array_equal(a["segments"], b["segments"])
    assert a["revenue"] == c["revenue"]                              # determinista


def test_higher_price_sells_fewer_units_same_customers():
    env = PricingEnv(CFG)
    low = run_episode(env, keep_policy, 5, 2)
    high = run_episode(env, keep_policy, 5, 9)
    env.reset(seed=5, options={"initial_level": 2})
    units = lambda level: sum(env.step(KEEP)[4]["units"][0] for _ in range(EPISODE_DAYS)) \
        if env.reset(seed=5, options={"initial_level": level}) else None
    assert units(2) > units(9)
    assert low["revenue"] != high["revenue"]


def test_oracle_beats_best_fixed_price():
    env = PricingEnv(CFG)
    best = best_fixed_levels(env)[0]
    seeds = range(40)
    fixed = np.mean([run_episode(env, keep_policy, s, best)["revenue"] for s in seeds])
    oracle = np.mean([run_episode(env, oracle_policy, s, best)["revenue"] for s in seeds])
    assert oracle > fixed * 1.05


def test_multi_product_env_steps():
    env = PricingEnv(CFG, PRODUCTS)
    env.reset(seed=0, options={"initial_level": [2, 5, 8]})
    _, _, _, _, info = env.step([MOVES.index(1), KEEP, MOVES.index(-2)])
    assert list(env.levels) == [3, 5, 6] and info["units"].shape == (3,)


# ---------- pantalla (sin ventana; agente sustituido por una política fija) ----------

class FixedAgent:
    def predict(self, obs, deterministic=True):
        return (KEEP if obs.shape[0] == 7 else np.array([KEEP] * 3)), None


def test_screen_runs_episode_and_returns_with_esc(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from common import ui
    from task_c_pricing import screen as screen_c

    screen = ui.create_window((1024, 640), "UrbanStyle")
    bundle = {"cfg": CFG, "summary": None, "agents": {k: FixedAgent() for k in ("PPO", "DQN", "PPO multi")}}
    view = screen_c.PricingScreen(bundle, size=screen.get_size())
    view.draw(screen)                                           # día 0
    view.update(0.1)
    for _ in range(EPISODE_DAYS + 5):
        view.step_day() if view.day < EPISODE_DAYS else view.update(0.1)
    assert view.day == EPISODE_DAYS
    # Mismos clientes: el agente fijo en el mismo precio que el mejor fijo gana exactamente lo mismo.
    view.price_slider.value = view.best_fixed[0]
    view.restart()
    for _ in range(10):
        view.step_day()
    assert view.cumulative("agent")[-1] == view.cumulative("fixed")[-1]
    view.mouse = view.revenue_plot.center
    view.draw(screen)                                           # con tooltip
    click = lambda b: view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=b.rect.center))
    click(view.mode_buttons[1])
    assert view.products == PRODUCTS and view.model_key == "PPO multi"
    view.step_day()
    view.draw(screen)

    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="", scancode=0))
    monkeypatch.setattr(screen_c, "load_bundle", lambda: bundle)
    screen_c.run(screen, pygame.time.Clock())
    pygame.quit()
