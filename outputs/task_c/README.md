# Salidas de la Tarea C (precios dinámicos con RL)

`training/` se genera con `python -m task_c_pricing.train`; `metrics/` y `figures/` con `python -m task_c_pricing.evaluate`. Montos en $ por episodio de 60 días.

## training/ — entrenamiento
- `dqn.monitor.csv`, `ppo.monitor.csv`, `ppo_multi.monitor.csv`: formato `Monitor` de stable-baselines3, una fila por episodio de entrenamiento: `r` = recompensa total (ingreso / 1000), `l` = largo (60 días), `t` = segundos desde el inicio.

## metrics/ — evaluación (200 episodios, semillas 10000–10199, distintas de las de entrenamiento)
- `fixed_prices_single.csv` / `fixed_prices_multi.csv`: ingreso medio y desviación con cada uno de los 11 precios fijos (50 %–150 % del precio de referencia), por producto. De aquí sale el **mejor precio fijo**.
- `comparison_single.csv` (solo Ropa) y `comparison_multi.csv` (tres productos): por política, ingreso medio, desviación, diferencia media contra el mejor precio fijo con IC 95 % (bootstrap pareado), % de ganancia y % de episodios en que gana.
- `episodes_single.csv`: ingreso de cada episodio y política (datos de la comparación pareada).
- `initial_price_single.csv`: ingreso de DQN y PPO según el precio con que empiezan (lo que controla el slider de la pantalla), frente al precio fijo en ese mismo precio y al mejor precio fijo.
- `policy_by_segment.csv`: precio medio que cobra cada política en cada segmento (× precio de referencia) frente al óptimo del segmento.
- `summary.json`: números clave que usa la pantalla (mejor precio fijo y resultados de cada política).

## figures/
- `fixed_price_curve.png`: ingreso según el precio fijo, con DQN, PPO y oráculo como líneas de referencia.
- `demand_by_segment.png`: la regla de demanda supuesta para Ropa: % de clientes que compra según el precio y ingreso esperado por segmento, con el óptimo de cada uno y el mejor precio fijo. `slides/demand_buy.png` y `slides/demand_revenue.png` son sus dos paneles por separado (cuadrados, para las diapositivas).
- `comparison_single.png`, `comparison_multi.png`: barras de ingreso medio ± desviación por política.
- `learning_curves.png`: ingreso por episodio durante el entrenamiento de los 3 agentes, con el mejor precio fijo y el oráculo.
- `episode_example.png`: un mismo episodio: precio de PPO, del mejor precio fijo y del oráculo según el segmento del día, e ingreso acumulado.
- `screen_task_c.png`, `screen_task_c_multi.png`: capturas de la pantalla (un producto con tooltip; tres productos al final del episodio).
