"""Tarea C (Sebastián): modelo de demanda del entorno de precios ("disposición a pagar").

El dataset solo registra compras: da ESCALAS reales, no cómo reaccionaría un cliente a otro precio. Por eso
la reacción al precio es un SUPUESTO de simulación, declarado aquí y en el informe:

  * Cada día llegan clientes interesados en cada producto: N ~ Poisson(SHOPPERS_PER_DAY)  [supuesto].
  * Cada cliente tiene una disposición a pagar (precio máximo) WTP ~ LogNormal con mediana
    wtp_scale[segmento][producto] × precio de referencia y dispersión WTP_SIGMA            [supuesto].
  * Compra si precio <= WTP; si compra, lleva q unidades (1–4) con la distribución REAL de su segmento.
  * Más precio -> menos compradores: la demanda baja con el precio (si no, el agente subiría al máximo).

Lo que sale del dataset (`common.data`):
  * precio de referencia de cada producto = precio medio pagado en su categoría (`pricing_stats`);
  * segmentos = 3 rangos de edad, con su participación en las transacciones y su distribución de cantidades.

Cada segmento tiene un precio óptimo distinto (supuesto: los adultos 36-55 pagan más por ropa, los jóvenes
por electrónica, los mayores por belleza). El segmento que predomina cambia de un día a otro (cadena de Markov):
por eso un precio fijo no puede ser siempre el mejor y tiene sentido ajustar el precio.

Todas las extracciones aleatorias de un día dependen solo de la semilla, NO del precio (números aleatorios
comunes): así RL, precio fijo y oráculo se comparan exactamente con los mismos clientes.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from common import config

PRODUCTS = ("Clothing", "Beauty", "Electronics")         # las 3 categorías originales del dataset
PRODUCT_NAMES = {"Clothing": "Ropa", "Beauty": "Belleza", "Electronics": "Electrónica"}
SEGMENTS = ("18-35", "36-55", "56-64")                   # rangos de edad (agrupando los de common.data)
SEGMENT_NAMES = {"18-35": "Jóvenes 18-35", "36-55": "Adultos 36-55", "56-64": "Mayores 56-64"}
AGE_BINS = (17, 35, 55, 64)
QUANTITIES = (1, 2, 3, 4)

# ---- Supuestos de simulación (no salen del dataset) ----
WTP_SCALE = {                 # mediana de la disposición a pagar / precio de referencia
    "18-35": {"Clothing": 0.95, "Beauty": 1.05, "Electronics": 1.65},
    "36-55": {"Clothing": 1.70, "Beauty": 1.30, "Electronics": 1.30},
    "56-64": {"Clothing": 1.30, "Beauty": 1.75, "Electronics": 0.95},
}
WTP_SIGMA = 0.30              # dispersión del log de la disposición a pagar
SHOPPERS_PER_DAY = 40.0       # clientes interesados por producto y día
STAY_PROB = 0.85              # probabilidad de que mañana predomine el mismo segmento
PRICE_MULTIPLIERS = tuple(round(0.5 + 0.1 * i, 1) for i in range(11))   # 50 % … 150 % del precio de referencia

CONFIG_PATH = config.MODELS_DIR / "pricing_config.json"


@dataclass(frozen=True)
class DemandConfig:
    """Todo lo necesario para reproducir la demanda; se guarda junto a los agentes (models/pricing_config.json)."""
    products: tuple[str, ...]
    ref_prices: tuple[float, ...]                 # por producto (dataset)
    segments: tuple[str, ...]
    segment_share: tuple[float, ...]              # participación de cada segmento (dataset)
    qty_probs: tuple[tuple[float, ...], ...]      # [segmento] -> P(cantidad = 1..4) (dataset)
    wtp_scale: tuple[tuple[float, ...], ...]      # [segmento][producto] (supuesto)
    wtp_sigma: float = WTP_SIGMA
    shoppers_per_day: float = SHOPPERS_PER_DAY
    stay_prob: float = STAY_PROB
    price_multipliers: tuple[float, ...] = PRICE_MULTIPLIERS

    # ---------- guardar / cargar ----------
    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "DemandConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        to_tuple = lambda v: tuple(to_tuple(x) for x in v) if isinstance(v, list) else v
        return cls(**{k: to_tuple(v) for k, v in raw.items()})

    def subset(self, products: tuple[str, ...]) -> "DemandConfig":
        """La misma demanda restringida a algunos productos (p. ej. solo Clothing para el entorno simple)."""
        idx = [self.products.index(p) for p in products]
        return DemandConfig(products, tuple(self.ref_prices[i] for i in idx), self.segments, self.segment_share,
                            self.qty_probs, tuple(tuple(row[i] for i in idx) for row in self.wtp_scale),
                            self.wtp_sigma, self.shoppers_per_day, self.stay_prob, self.price_multipliers)

    # ---------- precios ----------
    @property
    def n_levels(self) -> int:
        return len(self.price_multipliers)

    def price(self, product: int, level: int) -> float:
        return self.ref_prices[product] * self.price_multipliers[level]

    def level_of(self, product: int, price: float) -> int:
        """Nivel de la grilla de precios más cercano a `price`."""
        prices = np.array([self.price(product, k) for k in range(self.n_levels)])
        return int(np.argmin(np.abs(prices - price)))

    # ---------- demanda esperada (para el oráculo, las pruebas y el informe) ----------
    def mean_qty(self, segment: int) -> float:
        return float(np.dot(QUANTITIES, self.qty_probs[segment]))

    def buy_probability(self, price, segment: int, product: int):
        """P(WTP >= precio): decreciente en el precio."""
        median = self.wtp_scale[segment][product] * self.ref_prices[product]
        return ndtr(-(np.log(price) - np.log(median)) / self.wtp_sigma)

    def expected_units(self, price, segment: int, product: int):
        return self.shoppers_per_day * self.buy_probability(price, segment, product) * self.mean_qty(segment)

    def expected_revenue(self, price, segment: int, product: int):
        return price * self.expected_units(price, segment, product)

    def optimal_level(self, segment: int, product: int) -> int:
        """Nivel de precio que maximiza el ingreso esperado de un segmento (lo sabe solo el oráculo)."""
        revenue = [self.expected_revenue(self.price(product, k), segment, product) for k in range(self.n_levels)]
        return int(np.argmax(revenue))

    def reference_units(self, product: int) -> float:
        """Unidades esperadas por día al precio de referencia, promediando segmentos (para normalizar)."""
        return float(sum(share * self.expected_units(self.ref_prices[product], s, product)
                         for s, share in enumerate(self.segment_share)))

    # ---------- simulación de un día ----------
    def next_segment(self, rng: np.random.Generator, segment: int) -> int:
        """Cadena de Markov: se queda con prob. stay_prob; si no, se sortea según la participación real."""
        stay, draw = rng.random(), rng.choice(len(self.segments), p=self.segment_share)
        return segment if stay < self.stay_prob else int(draw)

    def sample_day(self, rng: np.random.Generator, segment: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Clientes del día por producto: (disposición a pagar, unidades que llevaría cada uno)."""
        day = []
        for p in range(len(self.products)):
            n = rng.poisson(self.shoppers_per_day)
            median = self.wtp_scale[segment][p] * self.ref_prices[p]
            wtp = median * np.exp(self.wtp_sigma * rng.standard_normal(n))
            qty = rng.choice(QUANTITIES, size=n, p=self.qty_probs[segment])
            day.append((wtp, qty))
        return day

    @staticmethod
    def units_sold(wtp: np.ndarray, qty: np.ndarray, price: float) -> int:
        return int(qty[wtp >= price].sum())


def build_config(df=None, products: tuple[str, ...] = PRODUCTS) -> DemandConfig:
    """Arma la configuración con las escalas del dataset (`df` = load_sales(); si es None se lee el CSV)."""
    from common.data import load_sales, pricing_stats

    df = load_sales() if df is None else df
    by_cat = pricing_stats(df, by=("Product Category",)).set_index("Product Category")
    ages = df.assign(Segment=np.select([df["Age"] <= AGE_BINS[1], df["Age"] <= AGE_BINS[2]], SEGMENTS[:2],
                                       SEGMENTS[2]))
    share = ages["Segment"].value_counts(normalize=True).reindex(SEGMENTS)
    qty = (ages.groupby("Segment")["Quantity"].value_counts(normalize=True).unstack()
           .reindex(index=SEGMENTS, columns=QUANTITIES, fill_value=0.0))
    return DemandConfig(
        products=tuple(products),
        ref_prices=tuple(round(float(by_cat.loc[p, "price_mean"]), 2) for p in products),
        segments=SEGMENTS,
        segment_share=tuple(round(float(x), 4) for x in share / share.sum()),
        qty_probs=tuple(tuple(round(float(x), 4) for x in row / row.sum()) for row in qty.to_numpy()),
        wtp_scale=tuple(tuple(WTP_SCALE[s][p] for p in products) for s in SEGMENTS),
    )


def load_config() -> DemandConfig:
    """La configuración guardada al entrenar; si aún no existe, se arma desde el CSV."""
    return DemandConfig.load() if CONFIG_PATH.exists() else build_config()
