# Tarea D: GAN para imágenes promocionales (notas para el informe)

> Documento de trabajo con la teoría, las decisiones (con números) y los resultados reales de la Tarea D.
> Cada número sale de un script del repositorio, indicado al lado. Todo corre en CPU (Python 3.13, torch 2.14, 10 hilos).

## 1. Requisitos (enunciado `UrbanStyle_Project_Plan-2026_2.pdf`)

**Objetivo:** crear imágenes promocionales de moda nuevas (se permite Fashion-MNIST).

| # | Requisito | Dónde se cumple | Evidencia |
|---|---|---|---|
| 1 | Entrenar una GAN muy pequeña (p. ej. DCGAN) sobre imágenes de producto | `task_d_gan/models.py`, `train.py` | `models/gan_generator.pt`; G 237 345 y D 36 641 parámetros |
| 2 | Generador: ruido → imagen 28×28 | `Generator` (z de 64 → 1×28×28, Tanh) | `test_generator_and_discriminator_shapes_and_range` |
| 3 | Discriminador: clasificar real/falso | `Discriminator` (logit + `BCEWithLogitsLoss`) | mismo test; curvas de D(x) y D(G(z)) |
| 4 | Entrenar unas pocas épocas y guardar las salidas | 20 épocas (§5.1) | `outputs/task_d/samples/`, `training/` (grilla por época y GIF) |
| Pygame | Botón para generar imágenes y slider de semilla del ruido | `task_d_gan/screen.py` | `outputs/task_d/figures/screen_task_d_*.png`, `test_screen_headless_flow` |
| Bonus | GAN condicional (cGAN) **y** póster promocional (se hicieron los dos) | `CondGenerator`/`CondDiscriminator`, `poster.py` | `cgan_confusion.png` (90,9 % de control), `poster/poster_seed_7.png` |

Rúbrica: 20 %, "Small GAN trained, sample outputs generated"; bonus: "Conditional GAN or promo poster". El PDF y `docs/UrbanStyle_Plan_de_Trabajo.md` (§5.4) coinciden en todos los puntos.

## 2. Datos

- **Fashion-MNIST** (Zalando): 60 000 imágenes de entrenamiento y 10 000 de prueba, 28×28 en escala de grises, 10 clases balanceadas (6000 por clase en entrenamiento). `torchvision` lo descarga solo a `data/FashionMNIST/`, que está en `.gitignore`.
- **Clases en español**, en el orden de las etiquetas 0–9: Camiseta, Pantalón, Suéter, Vestido, Abrigo, Sandalia, Camisa, Tenis, Bolso, Botín. `load_fashion_mnist` comprueba que el orden de torchvision coincide, con un `assert`.
- **Normalización a [-1, 1]** (`data.to_unit_range`: x/127,5 − 1), por la Tanh del generador (§3.5). Verificado: forma (60000, 1, 28, 28), mínimo −1, máximo 1. La inversa `to_uint8` recupera los 256 niveles de gris sin pérdida (`test_normalization_roundtrip_is_exact`).
- Todo el conjunto se carga en memoria como tensor, unos 188 MB en float32. Es más rápido que aplicar transforms imagen por imagen.
- Grilla de referencia: `outputs/task_d/samples/real_grid.png` (`python -m task_d_gan.data`).

## 3. Teoría (lo que hay que poder explicar)

### 3.1 La idea: dos redes que compiten
- **Generador G**: recibe ruido z ~ N(0, I) de 64 valores y devuelve una imagen 28×28. Nunca ve imágenes reales; aprende solo por el gradiente que le llega a través de D.
- **Discriminador D**: recibe una imagen y devuelve un número (logit). Si es positivo, la imagen "parece real". Es un clasificador binario real/falso.
- **Objetivo minimax** (Goodfellow et al., 2014; semana 7):

  min_G max_D  V(D, G) = E_x[log D(x)] + E_z[log(1 − D(G(z)))]

  D quiere acercar D(x) a 1 en las reales y D(G(z)) a 0 en las falsas. G quiere que D(G(z)) suba. En el óptimo teórico G reproduce la distribución de los datos y D(x) = 0,5 en todas partes (ya no puede distinguir).

### 3.2 Entropía cruzada binaria (BCE) y BCEWithLogitsLoss
- BCE(p, t) = −[t·log p + (1 − t)·log(1 − p)]. Con t = 1 vale −log p; con t = 0 vale −log(1 − p).
- El paso de D minimiza BCE(D(x), 1) + BCE(D(G(z)), 0) = −log D(x) − log(1 − D(G(z))). Es exactamente maximizar V.
- D devuelve un **logit**, sin sigmoide, y se usa `BCEWithLogitsLoss`, que aplica la sigmoide por dentro con el truco log-sum-exp. Si D tuviera una sigmoide y luego se usara `BCELoss`, un logit muy negativo daría sigmoid ≈ 0 y log(0) = −∞: gradientes inestables.

### 3.3 Pérdida no saturante del generador
- La pérdida "original" de G sería minimizar log(1 − D(G(z))). Al inicio D rechaza las falsas con facilidad (D(G(z)) ≈ 0). Ahí la curva log(1 − p) es casi plana y el gradiente para G es casi nulo: G no aprende. Eso es la **saturación**.
- La versión **no saturante** maximiza log D(G(z)), es decir, minimiza BCE(D(G(z)), 1) = −log D(G(z)). Tiene el mismo punto fijo, pero da gradientes grandes justo cuando G es malo. En código: `loss_G = bce(D(fake), ones)`, la "etiqueta 1 para las falsas" del taller.

### 3.4 `detach()` en el grafo computacional
- `fake = G(z)` construye un grafo que va de los pesos de G a la imagen. En el paso de D se usa `D(fake.detach())`: `detach()` devuelve el mismo tensor **desconectado** del grafo de G. Así `loss_D.backward()` calcula gradientes solo para D, y G no se toca ni se gasta cómputo en él.
- En el paso de G se usa `D(fake)` **sin** detach: el gradiente fluye a través de D, cuyos pesos no se actualizan porque solo se llama `opt_G.step()`, hasta los pesos de G.
- La prueba `test_detach_keeps_the_discriminator_step_away_from_G` lo comprueba: después del paso de D, todos los `grad` de G son `None`.

### 3.5 Tanh y datos en [-1, 1]
- G termina en `Tanh`, cuya salida está en [-1, 1]. Las imágenes reales se llevan a la misma escala: x/127,5 − 1 (`data.to_unit_range`).
- Si las reales estuvieran en [0, 1] y las falsas en [-1, 1], D las distinguiría por el rango de valores (cualquier píxel negativo delata una falsa) sin aprender nada de prendas. Sería una pista artificial.
- Para mostrar las imágenes se aplica la inversa exacta, (x + 1)·127,5 (`data.to_uint8`). La pantalla, el póster y la evaluación usan esas mismas dos funciones (prueba `test_normalization_roundtrip_is_exact`).

### 3.6 BatchNorm y LeakyReLU
- **BatchNorm** normaliza cada canal con la media y la varianza del lote y luego reescala con parámetros aprendidos. Estabiliza el entrenamiento de G y de D (DCGAN, Radford et al., 2016). Durante el entrenamiento usa las estadísticas del lote. En modo `eval()` usa las medias móviles guardadas.
  - Consecuencia práctica: para generar se usa `G.eval()`. Así la imagen de un z no depende de las demás imágenes del lote, y la misma semilla da siempre la misma imagen aunque se generen 1 o 100 (prueba `test_same_seed_same_images_and_batch_independence`).
  - Se midió que generar en modo eval no empeora la calidad (FID-CNN en §6.5).
- **LeakyReLU(0,2)** en D: para x < 0 deja pasar 0,2·x en lugar de 0. Con ReLU, las neuronas con entrada negativa dan gradiente 0. D necesita pasar gradiente útil hacia G para **todas** las imágenes, incluso las que rechaza con fuerza, así que se evita apagar neuronas. G usa ReLU, como en la DCGAN original.
- **ConvTranspose2d(k=4, s=2, p=1)** duplica el tamaño: (n − 1)·2 − 2·1 + 4 = 2n, de 7 a 14 y de 14 a 28. **Conv2d(k=4, s=2, p=1)** lo reduce a la mitad: de 28 a 14 y de 14 a 7.

### 3.7 Adam con betas = (0,5, 0,999)
- lr = 2·10⁻⁴ y β₁ = 0,5 (en lugar del 0,9 por defecto) vienen del paper DCGAN y del taller. Con β₁ menor, el momento "recuerda" menos los gradientes viejos. Eso importa porque el objetivo de cada red cambia en cada paso (el rival también aprende), y con mucho momento se oscila más.

### 3.8 cGAN (Mirza y Osindero, 2014)
- Ambas redes reciben además la etiqueta y: G(z, y) y D(x, y). El objetivo pasa a ser E[log D(x|y)] + E[log(1 − D(G(z|y)|y))].
- D ya no pregunta "¿es real?", sino "¿es una imagen real **de la clase y**?". Si G genera un buen pantalón cuando se le pidió un bolso, D lo rechaza porque no corresponde a la clase. Solo así G aprende a obedecer la etiqueta.
- En el paper original, la etiqueta entra como one-hot concatenado. Aquí se usa `nn.Embedding`, una tabla aprendida con un vector por clase. Es matemáticamente equivalente a one-hot seguido de una capa lineal, pero sin construir el one-hot.

## 4. Arquitectura

### 4.1 DCGAN: la del taller W2_DCGAN, sin cambios

| Red | Capa | Salida |
|---|---|---|
| **G** | z (64) → `Linear(64, 2·32·7·7)` → reshape | 64 × 7 × 7 |
| | `BatchNorm2d` + `ReLU` → `ConvTranspose2d(64, 32, 4, 2, 1)` | 32 × 14 × 14 |
| | `BatchNorm2d` + `ReLU` → `ConvTranspose2d(32, 1, 4, 2, 1)` → `Tanh` | 1 × 28 × 28 en [-1, 1] |
| **D** | `Conv2d(1, 32, 4, 2, 1)` → `LeakyReLU(0,2)` | 32 × 14 × 14 |
| | `Conv2d(32, 64, 4, 2, 1)` → `BatchNorm2d` → `LeakyReLU(0,2)` | 64 × 7 × 7 |
| | `Flatten` → `Linear(3136, 1)` | 1 logit (sin sigmoide) |

Parámetros: G = 237 345 y D = 36 641 (`count_params`). Es una GAN "muy pequeña", como pide el enunciado. G tiene ~6,5 veces más parámetros que D, igual que en el taller.

**Decisión: no cambiar la arquitectura ni la inicialización.**
- El paper DCGAN inicializa los pesos con N(0; 0,02). El taller no lo menciona, así que se dejó la inicialización por defecto de PyTorch.
- La evidencia de §5 muestra que no hizo falta: D no domina, no hay colapso y el FID-CNN baja de forma monótona.
- Cambiar algo sin necesidad solo agregaría cosas que explicar.

### 4.2 cGAN: extensión mínima de la misma red (Mirza y Osindero, 2014)

| Red | Qué cambia respecto a la DCGAN | Parámetros |
|---|---|---|
| **G(z, y)** | `Embedding(10, 10)`: un vector aprendido por clase, concatenado a z → `Linear(74, 3136)`. El resto es idéntico | 268 805 (+31 460) |
| **D(x, y)** | `Embedding(10, 784)` → reshape 1×28×28: un "mapa" aprendido por clase que entra como **2.º canal** de la imagen (`Conv2d(2, 32, …)`). El resto es idéntico | 44 993 (+8 352) |

**Decisión: cómo entra la etiqueta en D.**

| Alternativa | Por qué no / por qué sí |
|---|---|
| **Mapa 28×28 aprendido como canal extra** (elegida) | La extensión más pequeña que mantiene el D del taller tal cual: solo cambia `in_channels` de 1 a 2. Se explica en una frase: D ve la imagen junto con una "plantilla" de la clase. Funciona: 90,9 % de control (§6.3) |
| One-hot repetido en 10 canales 28×28 | Equivalente en la práctica (cada canal constante actúa como un sesgo por clase en la 1.ª convolución), pero 11 canales de entrada en lugar de 2 |
| Discriminador de proyección (Miyato y Koyama, 2018) | Mejor en datasets grandes, pero no se vio en clase y es más difícil de explicar línea por línea; no hacía falta |
| Etiqueta solo en G | No es una cGAN: D no castigaría a G por generar otra clase, y la etiqueta se ignoraría |

- En G, `nn.Embedding` es matemáticamente igual a one-hot (como en el paper) seguido de una capa lineal; solo evita construir el one-hot.
- Las etiquetas de las falsas son **las mismas del lote real**. Así la mezcla de clases que ve D es idéntica en reales y falsas, y D no puede usar la frecuencia de clases como pista.

## 5. Entrenamiento

Esquema del taller, con un código común para los dos modelos (`train.py::train`). En cada lote de 128:

```
fake = G(z)                                    # (cGAN: G(z, y))
loss_D = BCE(D(real), 1) + BCE(D(fake.detach()), 0) ;  opt_D.step()   # paso de D
loss_G = BCE(D(fake), 1)                       # no saturante       ;  opt_G.step()   # paso de G
```

- Adam(lr = 2·10⁻⁴, betas = (0,5, 0,999)) para ambos.
- Semilla 42 (`config.set_seed`) para la inicialización y el ruido.
- `torch.Generator` con semilla para el orden de los lotes.
- Ruido fijo (`fixed_inputs`): 64 vectores en la DCGAN; en la cGAN, 8 vectores repetidos en las 10 clases.

Por época se registra (`{modelo}_history.csv`):
- las pérdidas medias;
- D(x) y D(G(z)) (antes y después del paso de D);
- y, con la CNN auxiliar, el **FID-CNN** de 5000 muestras, la **entropía de clases** y, en la cGAN, el **% de control**.

### 5.1 Decisión: 20 épocas (exploración de 30, `python -m task_d_gan.train --explore 30`)

| Época | 1 | 5 | 10 | 15 | **20** | 25 | 30 |
|---|---|---|---|---|---|---|---|
| FID-CNN DCGAN | 293,9 | 132,1 | 82,5 | 61,6 | **49,1** | 46,1 | 41,2 |
| FID-CNN cGAN | 182,5 | 54,6 | 21,0 | 11,8 | **10,2** | 8,1 | 7,3 |
| Control cGAN | 68,8 % | 84,5 % | 89,1 % | 90,4 % | **91,0 %** | 91,3 % | 90,7 % |

- En la época 20 ya se logra el **97 %** (DCGAN) y el **98 %** (cGAN) de la mejora de FID-CNN de 30 épocas: (293,9 − 49,1)/(293,9 − 41,2) = 0,97.
- El control de la cGAN está estable en ~91 % desde la época 12.
- A ojo, las grillas del ruido fijo de las épocas 20 y 30 son casi iguales: las mismas prendas, la 30 apenas más limpia.
- Costo: 20 épocas toman **~6 min (DCGAN) + ~6,5 min (cGAN)** (media de las 6 corridas por modelo, §5.2); 30 épocas toman 9–10,5 + 11,5–12 min (las dos exploraciones). Es lo que pide el enunciado ("unas pocas épocas") en un equipo sin GPU.
- Curvas con la época elegida marcada: `training/{dcgan,cgan}_explore_curves.png`.

### 5.2 Tiempos en CPU (portátil, 10 hilos)

| Qué | Tiempo |
|---|---|
| DCGAN, 20 épocas (modelo final, clon limpio) | 372 s (18,6 s/época); las 5 semillas del §6.8: 324–388 s |
| cGAN, 20 épocas (modelo final, clon limpio) | 381 s (19,1 s/época); las 5 semillas del §6.8: 367–434 s |
| CNN auxiliar, 6 épocas | 88 s |
| Evaluación completa (`evaluate.py`) | ~20 s |
| Pantalla: dibujar un cuadro | ~1 ms (60 FPS) |
| Pantalla: regenerar 15 imágenes o la interpolación | ~30 ms |
| Pantalla: crear el póster | ~0,3 s |

### 5.3 Reproducibilidad

- La corrida final de 20 épocas da **exactamente** los mismos números que las primeras 20 épocas de la exploración (p. ej. FID-CNN 49,129 y 10,208 en la época 20 de ambas corridas): mismo código, misma semilla, CPU.
- La exploración de 30 épocas se corrió **dos veces** y las dos corridas coinciden en todas las métricas de todas las épocas (diferencia máxima 0,0).
- **Clon limpio:** `git clone` + venv nuevo + `pip install -r requirements.txt` + `pytest` (54 pasan, sin descargar datos ni entrenar) + `python -m task_d_gan.train`. Los pesos de G salen **idénticos** a los versionados en los dos modelos, y las grillas por época y los GIF son byte a byte iguales. `evaluate.py` en el clon regenera `summary.json` y todas las figuras sin diferencias. Los checkpoints de `models/` son los de ese entrenamiento en el clon (mismos pesos; tiempos medidos sin otros procesos en paralelo).
- Después de guardar, `train.py` recarga el checkpoint y comprueba que genera **las mismas imágenes** con la misma semilla (`recargado genera las mismas imágenes: True`, en los dos modelos).
- Checkpoints al estilo del taller: `{"G", "D", "nz", "base", "conditional", "epochs", "seed", "classes", "hparams", "train_seconds", "final_metrics"}`. Pesan 1,1 MB (DCGAN) y 1,3 MB (cGAN), así que se suben a Git. La CNN auxiliar (`models/fashion_cnn.pt`) pesa 1,7 MB.

### 5.4 Lectura de las curvas (`training/{dcgan,cgan}_curves.png`)

| | DCGAN (época 20) | cGAN (época 20) | Equilibrio teórico |
|---|---|---|---|
| D(x) medio | 0,709 | 0,575 | 0,5 |
| D(G(z)) medio | 0,291 | 0,425 | 0,5 |
| loss_D | 0,80 | 1,19 | 2·ln 2 ≈ 1,39 |
| loss_G | 1,56 | 0,98 | ln 2 ≈ 0,69 |

- **D va adelante pero no domina.** Si dominara, D(x) tendería a 1, D(G(z)) a 0 y loss_G crecería sin parar. En cambio, ambas medias quedan **estables** desde la época ~7.
- La cGAN está más cerca del equilibrio: la etiqueta le da a G información que hace la competencia más pareja.
- **Pérdidas planas, calidad que sigue mejorando:** desde la época 7 las pérdidas de la DCGAN casi no cambian (loss_G ≈ 1,55), pero el FID-CNN sigue bajando de 103 a 49. Es el mensaje del taller: las pérdidas de una GAN no miden calidad, porque cada una se mide contra un rival que también mejora.
- En el transitorio inicial (épocas 1–3), D le gana claramente a G (D(x) = 0,87 y D(G(z)) = 0,13 en la época 2). Ahí la pérdida no saturante es la que permite que G siga recibiendo gradiente.

## 6. Evaluación (`python -m task_d_gan.evaluate` → `outputs/task_d/evaluation/summary.json`)

Las muestras de evaluación usan una semilla (142) distinta de la del monitoreo del entrenamiento (43). Por eso el FID-CNN de esta sección, con 10 000 muestras, difiere un poco del de la tabla §5.1, que usa 5000.

### 6.1 El juez: CNN auxiliar (semana 11)

- Arquitectura: `Conv(32)-MaxPool-Conv(64)-MaxPool-Linear(128)-Dropout-Linear(10)`.
- Entrenamiento: 6 épocas, Adam 10⁻³, sobre las 60 000 reales en [-1, 1], la **misma escala** que producen los generadores.
- **Exactitud en prueba: 91,08 %** (88 s en CPU).
- Por clase: Pantalón 98,9 %, Bolso 98,2 %, Sandalia 97,6 %, Botín 97,1 %, Tenis 96,5 %, Vestido 92,8 %, Camiseta 89,0 %, Suéter 87,7 %, Abrigo 86,0 %, **Camisa 67,0 %**.
- Camisa es la clase difícil conocida de Fashion-MNIST: se confunde con camiseta, suéter y abrigo.
- Consecuencia importante: **ninguna medición hecha con este juez puede ser mucho mejor que él**. Por eso cada métrica se compara con el mismo juez aplicado a imágenes reales.

### 6.2 DCGAN: colapso de modos y diversidad

| Medida (10 000 muestras) | DCGAN | Reales de prueba |
|---|---|---|
| Clases con ≥ 2 % de las muestras | **10/10** | 10/10 |
| % por clase (mín.–máx.) | 7,7 % (Tenis) – 13,4 % (Botín) | 8,5 % – 10,8 % |
| Entropía de clases normalizada (1 = uniforme) | **0,993** | 0,999 |
| Distancia de variación total con la distribución real | 0,080 | 0 |
| Distancia L2 media entre pares (diversidad) | **22,85** | 22,65 |

- **Sin colapso de modos.** Las 10 clases aparecen, ninguna por debajo del 7,7 %, y las muestras son tan distintas entre sí como las reales.
- Si G colapsara, la entropía caería hacia 0 y la distancia por pares muy por debajo de 22,6.
- Este monitoreo se hizo **en cada época**: la entropía pasó de 0,89 (época 1) a > 0,99 desde la época 8 y nunca bajó.
- Evidencia: `dcgan_class_coverage.png` y `samples/dcgan_by_predicted_class.png`.

### 6.3 cGAN: ¿la clase pedida controla la prenda?

Se pidieron 1000 imágenes de cada clase y se clasificaron con la CNN (`cgan_confusion.png`, `cgan_control_by_class.png`):

| Clase | Camiseta | Pantalón | Suéter | Vestido | Abrigo | Sandalia | Camisa | Tenis | Bolso | Botín | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cGAN: % clasificado como la pedida | 93,2 | 99,4 | 93,6 | 93,0 | **74,4** | 95,4 | 68,5 | 96,5 | 98,3 | 97,0 | **90,9 %** |
| CNN en reales (referencia) | 89,0 | 98,9 | 87,7 | 92,8 | 86,0 | 97,6 | 67,0 | 96,5 | 98,2 | 97,1 | 91,1 % |

- **El condicionamiento funciona.** El 90,9 % de las imágenes es de la clase pedida, prácticamente la exactitud del propio juez en reales (91,1 %). Al azar sería 10 %.
- Camisa (68,5 %) no es un fallo de la cGAN: la CNN reconoce solo el 67 % de las camisas reales.
- **Único punto débil real: Abrigo** (74,4 % frente a 86,0 % en reales). El 12 % de los abrigos pedidos se clasifica como Suéter y otro 12 % como Camisa: prendas de manga larga muy parecidas a 28×28.
- **Sin colapso dentro de cada clase.** La distancia media por pares dentro de cada clase pedida es el **97,7 %** de la de las reales de esa clase, en promedio. Una cGAN puede obedecer la clase pero generar siempre la misma prenda; aquí no pasa.
- **z = estilo, y = prenda.** En `training/cgan_epochs/epoch_20.png` cada columna usa el mismo z en las 10 clases. El z de la columna 1 da prendas oscuras con estampado en todas las clases, y el de la columna 4 da prendas claras y lisas.

### 6.4 Memorización: ¿crea o copia?

Para 1000 imágenes de cada grupo se buscó la más cercana (L2 en píxeles) entre las **60 000** de entrenamiento:

| Grupo | Mediana | Percentil 1 | Mínimo | % por debajo del p1 de las reales nuevas |
|---|---|---|---|---|
| Reales **nuevas** (prueba): referencia de "no copia" | 6,99 | 3,08 | 1,67 | — |
| DCGAN | 10,08 | 4,68 | 4,03 | **0 %** |
| cGAN | 8,70 | 4,61 | 3,83 | **0 %** |

- **No hay memorización.** Las generadas están, en mediana, *más lejos* del entrenamiento que imágenes reales que la red nunca vio. Ninguna generada está tan cerca como el 1 % más cercano de las reales nuevas.
- Hay reales de prueba a 1,7 de una de entrenamiento (casi duplicados dentro de Fashion-MNIST). La generada más cercana está a 3,8.
- El peor caso se ve en `memorization_nearest_pairs.png`: las 10 generadas más parecidas a algo de entrenamiento, al lado de su vecina. Son casi todas pantalones, la silueta más simple del dataset, y difieren en brillo, textura y forma: son variaciones, no copias.

### 6.5 Calidad: FID-CNN (aproximación del FID, ver `metrics.py`)

**Qué es.** La distancia de Fréchet entre dos gaussianas ajustadas a los rasgos de 128 valores de la penúltima capa de la CNN auxiliar: ‖μ₁ − μ₂‖² + Tr(Σ₁ + Σ₂ − 2(Σ₁Σ₂)^½).

**No es el FID estándar.** Ese usa Inception-v3: 2048 rasgos de una red entrenada en ImageNet, a color y en 299×299. Usarlo aquí implicaría:
- descargar ~100 MB de pesos;
- ampliar imágenes de 28×28 en gris a 299×299 RGB;
- medir con una red que nunca vio Fashion-MNIST.

El FID-CNN es más barato y está adaptado al dominio, pero **solo sirve para comparar** entre sí modelos y épocas con el mismo extractor. Sus valores no son comparables con FID publicados. Por eso se reportan también el piso y el techo de la escala:

| Comparación (10 000 vs. 10 000 reales de prueba) | FID-CNN |
|---|---|
| Reales de entrenamiento (piso: "indistinguible") | 0,35 |
| **cGAN** | **10,41** |
| **DCGAN** | **47,92** |
| DCGAN generando con BatchNorm en modo entrenamiento (estadísticas del lote) | 47,97 |
| Ruido uniforme (techo: "nada que ver") | 639,8 |

- La **cGAN es ~4,6 veces mejor** que la DCGAN con casi la misma red. La etiqueta libera a G de "decidir" la clase dentro de z, y eso evita las formas híbridas (§6.6, §8).
- **Generar en modo `eval()` no empeora nada** (47,92 frente a 47,97). Por eso se puede usar eval, que garantiza el determinismo por semilla.

### 6.6 Reconocibles: % de imágenes que la CNN clasifica con confianza > 0,9

| Reales de prueba | cGAN | DCGAN |
|---|---|---|
| 74,7 % | 67,7 % | **40,0 %** |

Es la misma idea del Inception Score: una imagen buena es fácil de clasificar.
- La cGAN genera casi tan reconocible como lo real.
- La DCGAN produce bastantes **formas ambiguas**, a medio camino entre dos prendas (se ven en `samples/dcgan_samples.png`).
- La interpolación explica por qué (§6.7): el espacio z de la DCGAN no tiene fronteras de clase, así que muchos z caen "entre" dos prendas.

### 6.7 Interpolación latente (hito 5 del taller)

- `samples/dcgan_interpolation.png`: z(t) = (1 − t)·z₀ + t·z₁ en 10 pasos, 6 pares.
  - **Los cambios son graduales**: forma, brillo y largo varían poco a poco y no hay saltos bruscos.
  - Esa continuidad es evidencia de que G aprendió un espacio continuo y no una tabla de imágenes memorizadas.
  - Cuando z₀ y z₁ son de prendas distintas (fila z3→z4: pantalón → camisa; fila z9→z10: prenda → bolso), los pasos intermedios son **híbridos**.
- `samples/cgan_interpolation.png`: la misma interpolación con la **clase fija** en cada fila. Todos los pasos siguen siendo de la clase y solo cambia el estilo.
- En la pantalla, la vista "Interpolación" anima este camino entre la semilla s y la s + 1.

### 6.8 Robustez entre semillas (`python -m task_d_gan.seeds` → `evaluation/seed_results.csv`)

**Pregunta.** ¿Los resultados dependen de una semilla afortunada?

**Cómo se midió.**
- Se reentrenaron la DCGAN y la cGAN con **5 semillas más (1–5)**, con la misma configuración y las mismas 20 épocas.
- Cada modelo se evaluó **exactamente igual** que el final: mismas 10 000 muestras de evaluación y misma CNN. Comprobación: recalculadas con el mismo código, las métricas de la semilla 42 coinciden con `summary.json` (`semilla 42 = summary.json: True`).
- Cada entrenamiento tomó menos de 10 min en CPU (DCGAN 324–388 s; cGAN 367–434 s), así que se usaron 5 semillas y no 3.
- Los pesos de esas semillas son solo para evaluar: quedan en `checkpoints/task_d/seeds/`, fuera de Git, y **no reemplazan** los de `models/`.

**El modelo final no se eligió por ser el mejor.**
- Es el de `SEED = 42`, fijada de antemano en `common/config.py` desde el primer commit del proyecto (`6f90344`), antes de cualquier código de la Tarea D.
- Las 5 semillas del estudio son **distintas de 42**. Así la 42 se compara contra una distribución independiente; si estuviera incluida, caería en su propio rango por definición.

| Métrica (10 000 muestras) | Semilla 42 (final) | 5 semillas: media ± desv. | Rango de las 5 | ¿42 en el rango? | Puesto de 42 entre 6 |
|---|---|---|---|---|---|
| DCGAN · FID-CNN (menor es mejor) | 47,92 | 49,62 ± 2,17 | 46,70 – 52,44 | sí | 2.º |
| DCGAN · clases cubiertas (≥ 2 %) | 10/10 | 10/10 en las 5 | 10 – 10 | sí | — |
| DCGAN · entropía de clases | 0,993 | 0,994 ± 0,002 | 0,992 – 0,996 | sí | 5.º |
| DCGAN · clase menos generada | 7,7 % | 8,1 ± 0,3 % | 7,6 – 8,4 % | sí | 5.º |
| DCGAN · diversidad (L2 por pares; reales 22,65) | 22,85 | 22,63 ± 0,15 | 22,48 – 22,88 | sí | 2.º |
| DCGAN · reconocibles (confianza > 0,9) | 40,0 % | 39,0 ± 0,9 % | 38,2 – 40,4 % | sí | 2.º |
| cGAN · FID-CNN | 10,41 | 11,32 ± 1,13 | 9,98 – 12,93 | sí | 2.º |
| cGAN · **control global** | **90,9 %** | **90,2 ± 0,3 %** | 89,9 – 90,7 % | **no (+0,2 sobre el máx.)** | 1.º |
| cGAN · diversidad dentro de clase / real | 0,977 | 0,983 ± 0,011 | 0,973 – 0,995 | sí | 4.º |
| cGAN · diversidad (L2 por pares) | 22,144 | 22,47 ± 0,32 | 22,147 – 22,83 | **no (−0,003 bajo el mín.)** | 6.º |
| cGAN · reconocibles | 67,7 % | 66,6 ± 0,7 % | 65,5 – 67,2 % | **no (+0,5 sobre el máx.)** | 1.º |

Control de la cGAN **por clase**: media ± desviación de las 5 semillas (rango) y valor de la semilla 42:

| Clase | Camiseta | Pantalón | Suéter | Vestido | Abrigo | Sandalia | Camisa | Tenis | Bolso | Botín |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 semillas | 90,3 ± 1,8 | 99,6 ± 0,2 | 84,2 ± 4,3 | 94,1 ± 1,1 | 80,2 ± 3,8 | 96,9 ± 0,9 | 66,9 ± 4,8 | 93,3 ± 2,1 | 98,5 ± 0,5 | 97,6 ± 0,5 |
| Rango | 87,9–92,7 | 99,4–100 | 76,6–87,2 | 92,8–95,5 | 76,1–84,8 | 96,0–98,4 | 62,0–72,0 | 90,3–96,0 | 98,0–99,0 | 97,1–98,5 |
| Semilla 42 | 93,2 | 99,4 | 93,6 | 93,0 | 74,4 | 95,4 | 68,5 | 96,5 | 98,3 | 97,0 |

**Lectura (declarada en el informe).**
- **La calidad es consistente entre semillas.**
  - Las 6 DCGAN cubren 10/10 clases, con entropía ≥ 0,992 y FID-CNN entre 46,7 y 52,4.
  - Las 6 cGAN obedecen la clase pedida en un 89,9–90,9 %, con FID-CNN entre 10,0 y 12,9.
  - La diferencia cGAN vs. DCGAN (≈ 11 frente a ≈ 50) y la conclusión "sin colapso" se cumplen en **todas** las semillas.
  - Grillas con el mismo ruido fijo: `seed_samples_dcgan.png` y `seed_samples_cgan.png`. En la cGAN, cada columna es su clase en las 6 filas. El mismo z da un estilo distinto en cada corrida, porque cada entrenamiento aprende su propio mapeo z → imagen.
- **La semilla 42 cae dentro del rango de las otras 5 en 8 de 11 métricas** y fuera en 3, todas de la cGAN y por márgenes pequeños:
  - control +0,2 puntos sobre el máximo (z = +2,4);
  - reconocibles +0,5 sobre el máximo (z = +1,7);
  - diversidad −0,003 bajo el mínimo (z = −1,0).
- **Por qué esto no indica que la 42 sea especial.** Si las 6 semillas son intercambiables, la probabilidad de que una quede en un extremo de 6 valores es 2/6 = 1/3 por métrica. En 11 métricas se esperan ~3,7 extremos y se observan 3. Además, en dos de las métricas la 42 es la mejor y en una la peor.
- **Qué se reporta.** El control de 90,9 % es el del modelo que usa la app. La expectativa para un entrenamiento nuevo es **90,2 ± 0,3 %**: un 0,7 puntos menor, una diferencia sin importancia práctica.
- **Por clase hay bastante variación entre semillas** (Suéter 76,6–93,6 %; Camisa 62–72 %). Las cifras por clase de un solo modelo no deben sobreinterpretarse.
  - Lo que sí se sostiene en las 6 semillas: **Abrigo** es el punto débil (74–85 %, siempre bajo la referencia de 86 % de la CNN), y **Camisa** queda limitada por el juez (62–72 %, frente a 67 % de la CNN en reales).
  - En Abrigo la 42 es la peor de las 6 (74,4 %); en Suéter, la mejor (93,6 %).

## 7. Póster promocional (bonus 2, `python -m task_d_gan.poster --seed 7`)

- `outputs/task_d/poster/poster_seed_7.png`, de 1200×1500 (4:5).
- **Composición** con el estilo editorial de la app: fondo #0F0E0D, titular "La colección / *que nadie dibujó.*" en DM Serif Display, acento #FF5A36. Una prenda protagonista (Botín), una cuadrícula de 2×2 y una fila de 5: las 10 clases, numeradas como en un catálogo.
- **Nota visible** al pie: "Prendas generadas por IA (cGAN entrenada en Fashion-MNIST) · Imágenes de 28×28 px ampliadas · semilla del ruido 7". La bajada dice "Ninguna existía antes de este póster". Queda claro que **no son fotos de productos reales**.
- **Curaduría automática y reproducible:** por clase, 16 candidatas de la cGAN. Entre las que la CNN reconoce con confianza > 0,9 se toma el **medoide** en el espacio de rasgos de la CNN (la candidata más "típica").
  - Se probó primero "la de mayor confianza" y elegía muestras atípicas (un vestido con una manga asimétrica, una sandalia hecha de puntos): lo más *distintivo* para el clasificador, no lo más *representativo*.
- **Ampliación con Lanczos.** Se compararon nearest (deja escalones de píxel), bicúbica (más borrosa), Lanczos y bicúbica + unsharp (crea halos blancos). Lanczos es la más nítida sin artefactos.
- **Tinte y niveles.** El negro del fondo pasa al tono del póster y el blanco al marfil de la marca. El ajuste de niveles por prenda se aplica **solo en el póster**: la pantalla muestra la salida del generador sin retocar, para no maquillar la evidencia.
- La protagonista se amplía ×18,6. A esa escala un 28×28 se ve con "foco suave": es inevitable y lo declara la nota del pie.
- La semilla 7 se eligió a ojo para el PNG del informe, como haría un diseñador. `poster_seed_42.png` muestra otra semilla, con prendas más oscuras y texturadas. Desde la pantalla, "Crear póster" funciona con cualquier semilla.

## 8. Pantalla Pygame (`task_d_gan/screen.py`, `run(screen, clock)`)

- **Lenguaje editorial de la Tarea B** (`common/ui.py`): eyebrow de sección, titular en serif que **dice lo que se ve** ("15 vestidos nuevos, pedidos a la cGAN"), líneas finas, sin tarjetas, un acento. Los textos tienen ≥ 14 px lógicos. Capturas a escala 1,5 (1536×960) y al 75 % en `outputs/task_d/figures/`.
- **Controles:**
  - selector DCGAN / cGAN;
  - **slider de semilla** (0–999, también con ← →);
  - **botón "Generar"** (también con G): salta a una semilla nueva, que muestra el slider, con una aparición escalonada; volver a esa semilla da **las mismas imágenes**;
  - selector de prenda (las 10 clases por nombre, también con ↑ ↓) en la cGAN;
  - vistas Grilla / Interpolación;
  - "Crear póster" (también con P): compone el póster con la semilla actual, lo muestra y lo guarda en PNG, con un aviso que dice la ruta;
  - ESC: cierra el póster o vuelve al menú.
- **Evidencia en pantalla.** Debajo de cada imagen va la prenda que reconoce la CNN; en la cGAN se marca en dorado si no coincide con la pedida. En el panel lateral: "10/10 clases cubiertas, cada una entre el 8 % y el 13 %" (DCGAN), o "91 % son de la clase pedida" con el dato de la prenda elegida frente a la referencia de la CNN (cGAN). Todo se lee de `summary.json`.
- **Solo carga, nunca entrena.** Si faltan los modelos, muestra un mensaje con el comando para entrenar.
- **Determinismo.** `generate_from_seed` toma z de un `torch.Generator` con la semilla (no toca el estado aleatorio global) y genera con G en `eval()` (§3.6).
- **Normalización consistente.** La pantalla convierte [-1, 1] → 0–255 con la misma `to_uint8` que las grillas y la evaluación.
- **Rendimiento.**
  - Dibujar un cuadro toma ~1 ms (60 FPS).
  - Regenerar toma ~30 ms. Al principio la interpolación ampliaba sus 48 cuadros de golpe (550 ms, el slider se trababa); ahora cada cuadro se amplía la primera vez que se muestra.
- La tarjeta de la Tarea D en el menú ya no dice "en desarrollo".

## 9. Limitaciones

- **28×28 en escala de grises.** Ninguna imagen es fotorrealista ni a color. Ampliadas se ven suaves, porque no hay detalle que recuperar. El póster lo declara.
- **La DCGAN produce formas ambiguas.** Solo el 40 % de sus muestras se reconoce con confianza > 0,9, frente al 75 % de las reales. Su espacio z no separa las clases. Para uso promocional conviene la cGAN (68 %).
- **Abrigo en la cGAN** (74 % de control frente a 86 % de la CNN en reales): se confunde con suéter y camisa.
- **Las métricas dependen de la CNN auxiliar** (91 %). El FID-CNN no es comparable con el FID estándar de Inception. El control de clases de Camisa está limitado por el propio juez (67 % en reales).
- **Variabilidad entre semillas** (§6.8): las conclusiones se sostienen en las 6 semillas, pero el control por clase varía bastante entre corridas (p. ej. Suéter 77–94 %). Cinco semillas dan una idea de la dispersión, no un intervalo de confianza preciso.

## 10. Preguntas probables del profesor (con nuestra evidencia)

**¿Por qué D debe mejorar, pero sin dominar?**
- G aprende *solo* a través del gradiente de D. Si D es malo, el gradiente no dice hacia dónde mejorar.
- Si D es perfecto, D(G(z)) ≈ 0 en todas partes. Con la pérdida original, log(1 − D(G(z))) queda plana y el gradiente se anula: G deja de aprender.
- Aquí D va adelante, pero estable: D(x) = 0,71 y D(G(z)) = 0,29 en la DCGAN; 0,58 y 0,42 en la cGAN, desde la época ~7 (§5.4).

**¿Qué hace `detach()` en el grafo computacional?**
- Corta el tensor `fake` del grafo que lo conecta con los pesos de G. En el paso de D, `loss_D.backward()` solo calcula gradientes para D: G no se modifica y no se gasta cómputo en él.
- En el paso de G se usa `fake` sin detach, para que el gradiente atraviese D y llegue a G.
- Prueba: `test_detach_keeps_the_discriminator_step_away_from_G`. Después del paso de D, todos los `grad` de G son `None`.

**¿Por qué las pérdidas oscilan aunque la calidad mejore?**
- Cada pérdida se mide contra un rival que también cambia. Si G mejora, D también, y la pérdida de G no tiene por qué bajar.
- Es un juego, no una minimización: el equilibrio es un punto de silla, no un mínimo.
- Evidencia: desde la época 7 loss_G de la DCGAN se queda en ~1,55, mientras el FID-CNN baja de 103 a 49 y las grillas se ven cada vez mejor (`dcgan_progress.gif`). Por eso la calidad se juzga **viendo las muestras y con métricas externas**, no con las pérdidas.

**¿Qué es el colapso de modos y cómo se detecta con evidencia?**
- G descubre unas pocas salidas que engañan a D y genera solo esas: por ejemplo, solo pantalones, o el mismo pantalón con cualquier z.
- Se detecta con evidencia medible, no a ojo:
  1. **Cobertura de clases** con un clasificador. La DCGAN cubre 10/10, entre 7,7 % y 13,4 %; entropía 0,993 frente a 0,999 en reales, medida en cada época.
  2. **Diversidad** por pares. 22,85 frente a 22,65 en reales; dentro de cada clase, la cGAN tiene el 97,7 % de la diversidad real.
  3. **Ruido fijo** a lo largo de las épocas: las columnas siguen siendo distintas.
  4. **FID**, que penaliza la falta de varianza.

**¿En qué se diferencia el ruido z del código latente de un autoencoder?**
- En un autoencoder, el código latente lo produce un **encoder** a partir de una imagen concreta: cada código "es" una imagen y no hay una distribución fija de la cual muestrear.
- En una GAN no hay encoder. z se **muestrea** de una distribución elegida de antemano, N(0, I) de 64 valores, y G aprende a convertir *cualquier* z en una imagen plausible. Por eso se puede generar sin imagen de entrada y dar una semilla para obtener prendas nuevas.
- La interpolación entre dos z produce prendas intermedias plausibles (§6.7). En un autoencoder básico, los puntos entre dos códigos no tienen garantía de decodificar a algo plausible.

**¿Cómo sabemos que la cGAN no ignora la etiqueta?**
- Control del 90,9 % frente a 10 % al azar, con 1000 imágenes por clase (§6.3).
- Con el mismo z y distinta clase salen prendas distintas (`test_conditional_models_accept_the_10_classes` y la grilla de la cGAN).

**¿No estará copiando las imágenes de entrenamiento?**
- No. Ninguna de 1000 generadas por modelo está tan cerca de una imagen de entrenamiento como el 1 % más cercano de imágenes reales nuevas, y en mediana están más lejos (§6.4).

**¿No eligieron la semilla que daba mejores resultados?**
- No. `SEED = 42` está fijada en `common/config.py` desde el primer commit del proyecto, antes de escribir la Tarea D.
- Se reentrenó con 5 semillas más, distintas de 42 (§6.8). La 42 cae dentro del rango en 8 de 11 métricas, y en las 3 restantes por márgenes mínimos; se esperaban ~3,7 extremos por puro azar.
- Se reportan las dos cosas: el control del modelo de la app (90,9 %) y la expectativa para un entrenamiento nuevo (90,2 ± 0,3 %).

**¿Por qué [-1, 1] y Tanh? ¿Por qué un logit sin sigmoide?** Ver §3.5 y §3.2.

**¿Por qué 20 épocas?** Ver §5.1: el 97–98 % de la mejora de 30 épocas, a 2/3 del costo, con el control de la cGAN estable desde la época 12.

## 11. Cómo reproducir

```bash
python -m task_d_gan.data            # grilla de reales (descarga Fashion-MNIST la primera vez)
python -m task_d_gan.classifier      # CNN auxiliar (~1,5 min) -> models/fashion_cnn.pt
python -m task_d_gan.train           # DCGAN + cGAN, 20 épocas cada una (~13 min en CPU)
python -m task_d_gan.evaluate        # evaluación completa (~20 s) -> outputs/task_d/evaluation/
python -m task_d_gan.poster --seed 7 # póster -> outputs/task_d/poster/
python -m task_d_gan.train --explore 30   # (opcional, ~21 min) exploración para elegir las épocas
python -m task_d_gan.seeds              # (opcional, ~65 min) robustez: 5 semillas más (§6.8)
python main.py                       # menú -> "Tarea D"
```

## 12. Referencias

- Goodfellow et al. (2014). *Generative Adversarial Nets*. NeurIPS. Objetivo minimax y pérdida no saturante.
- Radford, Metz y Chintala (2016). *Unsupervised Representation Learning with Deep Convolutional GANs* (DCGAN). ICLR.
- Mirza y Osindero (2014). *Conditional Generative Adversarial Nets*. arXiv:1411.1784.
- Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS. Definición del FID.
- Xiao, Rasul y Vollgraf (2017). *Fashion-MNIST*. arXiv:1708.07747.
- Material del curso: taller W2_DCGAN (semana 10), Generative AI (semana 7), CNN (semana 11).
