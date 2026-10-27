# Salidas de la Tarea D (GAN para imágenes promocionales)

Todo se regenera con los scripts de `task_d_gan/` (ver el README principal). Las imágenes son de 28×28 en escala de grises (Fashion-MNIST); en las grillas se amplían sin interpolar para ver los píxeles reales.

## samples/: muestras generadas (lo que pide la rúbrica: "sample outputs generated")
- `real_grid.png`: 10 imágenes reales de entrenamiento por clase (referencia visual). `python -m task_d_gan.data`
- `real_test_grid.png`: 10 reales de **prueba** por clase, para comparar fila a fila con `cgan_by_class.png`.
- `dcgan_samples.png`: 100 muestras al azar de la DCGAN (sin elegir las mejores).
- `dcgan_by_predicted_class.png`: muestras de la DCGAN agrupadas por la prenda que reconoce la CNN auxiliar. Si una fila tiene huecos negros, la DCGAN genera poco esa clase.
- `cgan_by_class.png`: 10 muestras de la cGAN por clase **pedida** (fila = clase pedida).
- `dcgan_interpolation.png`: interpolación lineal entre dos vectores de ruido z (10 pasos, 6 pares). Es el hito 5 del taller.
- `cgan_interpolation.png`: la misma interpolación de z con la **clase fija** en cada fila: cambia el estilo y no la prenda.

## training/: entrenamiento
- `dcgan_history.csv` / `cgan_history.csv`: una fila por época:
  - `loss_D` y `loss_G`: pérdida BCE media;
  - `D_x`: media de D(x) en reales;
  - `D_G_z_before`: D(G(z)) en el paso de D;
  - `D_G_z_after`: D(G(z)) en el paso de G, con D ya actualizado;
  - `fid_cnn`: FID-CNN de 5000 muestras contra las 10 000 reales de prueba;
  - `class_entropy`: entropía normalizada de las clases generadas (1 = las 10 por igual);
  - `control_acc` (solo cGAN): fracción de imágenes que la CNN clasifica como la clase pedida;
  - `seconds`: tiempo acumulado en CPU.
- `dcgan_curves.png` / `cgan_curves.png`: pérdidas por iteración, medias de D(x) y D(G(z)) por época, y FID-CNN por época.
- `dcgan_epochs/epoch_XX.png` / `cgan_epochs/epoch_XX.png`: grilla del **ruido fijo** en cada época. En la cGAN, cada columna usa el mismo z en las 10 clases.
- `dcgan_progress.gif` / `cgan_progress.gif`: animación del progreso con esas grillas.
- `dcgan_explore_history.csv` / `dcgan_explore_curves.png` (y los de `cgan_`): corrida exploratoria de 30 épocas con la que se eligió el número de épocas (la línea discontinua marca la época 20). Sus pesos no se guardan.

## evaluation/: evaluación con evidencia (`python -m task_d_gan.evaluate`)
- `summary.json`: todos los números de la evaluación. La pantalla Pygame lee aquí la cobertura y el control.
- `dcgan_class_coverage.png`: % de muestras de la DCGAN por clase predicha, frente a lo que la CNN predice en reales. Sirve para detectar el colapso de modos.
- `cgan_confusion.png`: matriz clase pedida vs. clase predicha de la cGAN (1000 por clase).
- `classifier_confusion_real.png`: la misma matriz para la CNN sobre reales de prueba. Es la referencia del juez: la medición del control no puede ser mucho mejor que la propia CNN.
- `cgan_control_by_class.png`: por clase, el control de la cGAN frente a la exactitud de la CNN en reales.
- `memorization_nn_distance.png`: distancia de cada imagen a su vecina más cercana de las 60 000 de entrenamiento, en tres grupos: reales nuevas (prueba), DCGAN y cGAN.
- `memorization_nearest_pairs.png`: las 10 muestras de cada GAN **más parecidas** a una imagen de entrenamiento (el peor caso), junto a esa imagen.
- `seed_results.csv`: robustez entre semillas (`python -m task_d_gan.seeds`). Tiene una fila por modelo y semilla de entrenamiento: 42 (el modelo final) y 1–5 (solo para evaluar), con las mismas métricas de `summary.json` (FID-CNN, cobertura, control global y por clase, diversidad, reconocibles). Al final hay filas con la media y la desviación estándar de las 5 semillas, sin la 42.
- `seed_final_vs_others.csv`: para cada métrica, el valor de la semilla 42 frente al rango, la media ± desviación, el puntaje z y el puesto entre las 6.
- `seed_metrics.png`: un panel por métrica con las 5 semillas (gris) y la 42 (naranja).
- `seed_samples_dcgan.png` / `seed_samples_cgan.png`: el mismo ruido fijo generado por cada semilla de entrenamiento (fila = semilla). En la cGAN, las 10 clases con un mismo z.

## poster/: póster promocional (bonus)
- `poster_seed_7.png`: póster de UrbanStyle (1200×1500, formato 4:5), el destacado para el informe. Tiene una prenda por clase generada por la cGAN y la nota visible "Prendas generadas por IA". `python -m task_d_gan.poster --seed 7`
- `poster_seed_42.png`: el mismo póster con la semilla 42 (otro "estilo" de z: prendas más oscuras y texturadas).
- El botón "Crear póster" de la pantalla guarda aquí `poster_seed_<semilla>.png`.
- Cada prenda es la más típica (medoide en los rasgos de la CNN) entre 16 candidatas que la CNN reconoce con confianza > 0,9, con un ajuste de niveles solo para el póster (ver `task_d_gan/poster.py`).

## figures/: capturas de la pantalla Pygame (escala 1,5 = 1536×960; los `_75` son las mismas al 75 %)
- `screen_task_d_dcgan.png`: vista Grilla con la DCGAN (semilla 42). Debajo de cada prenda, la clase que reconoce la CNN; en el panel, la cobertura de clases.
- `screen_task_d_cgan.png`: la cGAN con la prenda "Vestido" pedida y el control de clase en el panel.
- `screen_task_d_interpolation.png`: vista Interpolación (tira de 10 pasos de z y animación) con la clase fija.
- `screen_task_d_poster.png`: "Crear póster": el póster mostrado sobre la pantalla y el aviso con la ruta del PNG.
- `menu.png`: menú principal con la Tarea D ya activa (sin "en desarrollo").
