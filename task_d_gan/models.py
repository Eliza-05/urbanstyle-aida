"""Tarea D (Elizabeth): generador y discriminador de la DCGAN (taller W2_DCGAN) y su extensión condicional (cGAN).

DCGAN (arquitectura del taller del profesor, semana 10):
  G: z (nz) -> Linear -> 7×7×(2·base) -> BN+ReLU -> ConvT -> 14×14×base -> BN+ReLU -> ConvT -> 28×28×1 -> Tanh
  D: 28×28×1 -> Conv -> 14×14×base -> LeakyReLU -> Conv -> 7×7×(2·base) -> BN -> LeakyReLU -> Linear -> 1 LOGIT
  D devuelve un logit (sin sigmoide) porque se entrena con BCEWithLogitsLoss, que aplica la sigmoide por dentro
  de forma numéricamente estable (log-sum-exp).

cGAN (Mirza & Osindero, 2014): lo único que cambia es que G y D también reciben la etiqueta de clase y.
  G: concat(z, embedding(y)) -> el mismo cuerpo de la DCGAN.
  D: la etiqueta se convierte en un mapa 28×28 aprendido (embedding -> 784 valores) y entra como 2.º canal
     de la imagen; el resto es el mismo D. Así D juzga "¿es una imagen real DE ESTA clase?", y G solo lo engaña
     si genera la prenda pedida.
"""
import torch
from torch import nn

NZ = 64      # dimensión del ruido z (taller: nz=64)
BASE = 32    # canales base (taller: base=32)


# ---------------------------------------------------------------------------
# DCGAN (obligatorio): exactamente la arquitectura del taller
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Ruido z (B, nz) -> imagen (B, 1, 28, 28) en [-1, 1]."""

    def __init__(self, nz: int = NZ, base: int = BASE, in_features: int | None = None):
        super().__init__()
        self.nz, self.base = nz, base
        # in_features permite que la cGAN reutilice este cuerpo con una entrada más ancha (z + etiqueta).
        self.fc = nn.Linear(in_features or nz, base * 2 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),   # 7×7 -> 14×14
            nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, 1, 4, 2, 1),          # 14×14 -> 28×28
            nn.Tanh(),                                      # salida en [-1, 1], como los datos
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.base * 2, 7, 7)       # "reshape" del vector a un mapa 7×7
        return self.net(x)


class Discriminator(nn.Module):
    """Imagen (B, 1, 28, 28) -> logit (B,): > 0 significa "parece real"."""

    def __init__(self, base: int = BASE, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),          # 28×28 -> 14×14
            nn.LeakyReLU(0.2, True),                        # pendiente 0.2 para x<0: el gradiente no muere
            nn.Conv2d(base, base * 2, 4, 2, 1),             # 14×14 -> 7×7
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(base * 2 * 7 * 7, 1),                 # LOGIT (sin sigmoide: BCEWithLogitsLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ---------------------------------------------------------------------------
# cGAN (bonus): la misma red + la etiqueta de clase
# ---------------------------------------------------------------------------

class CondGenerator(Generator):
    """(z, y) -> imagen de la clase y. La etiqueta entra como un vector aprendido (embedding) concatenado a z."""

    def __init__(self, n_classes: int = 10, nz: int = NZ, base: int = BASE, emb_dim: int = 10):
        super().__init__(nz, base, in_features=nz + emb_dim)
        self.n_classes = n_classes
        # Embedding = tabla aprendida de un vector por clase; equivale a one-hot (como en Mirza & Osindero)
        # seguido de una capa lineal, pero sin construir el one-hot.
        self.label_emb = nn.Embedding(n_classes, emb_dim)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.cat([z, self.label_emb(y)], dim=1))


class CondDiscriminator(Discriminator):
    """(imagen, y) -> logit de "real Y de la clase y". La etiqueta es un 2.º canal 28×28 aprendido."""

    def __init__(self, n_classes: int = 10, base: int = BASE):
        super().__init__(base, in_channels=2)
        self.n_classes = n_classes
        self.label_map = nn.Embedding(n_classes, 28 * 28)  # un "mapa" 28×28 aprendido por clase

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        label_channel = self.label_map(y).view(-1, 1, 28, 28)
        return super().forward(torch.cat([x, label_channel], dim=1))


# ---------------------------------------------------------------------------
# Generación determinista y checkpoints
# ---------------------------------------------------------------------------

def is_conditional(G: nn.Module) -> bool:
    return isinstance(G, CondGenerator)


def noise(seed: int, n: int, nz: int = NZ) -> torch.Tensor:
    """Ruido z ~ N(0, I) que depende SOLO de la semilla (generador propio: no altera la semilla global)."""
    return torch.randn(n, nz, generator=torch.Generator().manual_seed(int(seed)))


@torch.no_grad()
def generate(G: nn.Module, z: torch.Tensor, labels=None) -> torch.Tensor:
    """Imágenes en [-1, 1] a partir de z (y de las etiquetas, si G es condicional).

    G se usa en modo eval: BatchNorm usa sus estadísticas guardadas (running mean/var) en lugar de las del
    lote. Así la imagen de un z no depende de cuántas imágenes se generen a la vez (determinismo por semilla).
    """
    was_training = G.training
    G.eval()
    if is_conditional(G):
        if labels is None:
            raise ValueError("la cGAN necesita las etiquetas de clase")
        labels = torch.as_tensor(labels, dtype=torch.long)
        if labels.ndim == 0:                    # una sola clase para todas las imágenes
            labels = labels.expand(len(z))
        out = G(z, labels)
    else:
        out = G(z)
    G.train(was_training)
    return out


def generate_from_seed(G: nn.Module, seed: int, n: int, labels=None) -> torch.Tensor:
    """La misma semilla siempre produce las mismas n imágenes (lo que usa el slider de la pantalla)."""
    return generate(G, noise(seed, n, G.nz), labels)


def interpolate(z0: torch.Tensor, z1: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Interpolación lineal en el espacio latente (hito 5 del taller): (steps, nz) de z0 a z1, extremos incluidos."""
    t = torch.linspace(0, 1, steps).unsqueeze(1)
    return (1 - t) * z0.view(1, -1) + t * z1.view(1, -1)


def save_checkpoint(path, G, D, **info) -> None:
    """Estilo del taller: pesos de G y D + la configuración necesaria para reconstruirlos, en un solo archivo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "nz": G.nz, "base": G.base,
                "conditional": is_conditional(G), **info}, path)


def load_generator(path):
    """Reconstruye el generador (en modo eval) a partir del checkpoint. Devuelve (G, checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["conditional"]:
        G = CondGenerator(len(ckpt["classes"]), ckpt["nz"], ckpt["base"])
    else:
        G = Generator(ckpt["nz"], ckpt["base"])
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G, ckpt


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
