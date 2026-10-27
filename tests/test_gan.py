"""Pruebas rápidas de la Tarea D (no entrenan ni descargan Fashion-MNIST; las de modelos guardados se saltan si
no existen los checkpoints)."""
import importlib

import numpy as np
import pygame
import pytest
import torch
from torch import nn

from task_d_gan.data import CLASSES, to_uint8, to_unit_range
from task_d_gan.metrics import class_entropy, fid, mean_pairwise_distance, nearest_neighbors
from task_d_gan.models import (
    CondDiscriminator,
    CondGenerator,
    Discriminator,
    Generator,
    generate,
    generate_from_seed,
    interpolate,
    load_generator,
    noise,
    save_checkpoint,
)
from task_d_gan.paths import CGAN_CKPT, EVAL_JSON, GAN_CKPT

needs_models = pytest.mark.skipif(not (GAN_CKPT.exists() and CGAN_CKPT.exists()),
                                  reason="faltan los modelos entrenados (python -m task_d_gan.train)")


def test_gan_importable():
    assert importlib.import_module("task_d_gan.models") is not None


# ---------- datos ----------

def test_normalization_roundtrip_is_exact():
    """[0, 255] -> [-1, 1] -> [0, 255] sin perder ningún nivel de gris (misma fórmula en train y en pantalla)."""
    raw = torch.arange(256, dtype=torch.uint8).view(1, 16, 16)
    x = to_unit_range(raw)
    assert x.shape == (1, 1, 16, 16) and x.min() == -1 and x.max() == 1
    assert np.array_equal(to_uint8(x), raw.numpy())


def test_spanish_class_names():
    assert CLASSES == ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
                       "Sandalia", "Camisa", "Tenis", "Bolso", "Botín"]


# ---------- arquitectura ----------

def test_generator_and_discriminator_shapes_and_range():
    G, D = Generator(), Discriminator()
    img = G(torch.randn(8, G.nz))
    assert img.shape == (8, 1, 28, 28)
    assert img.min() >= -1 and img.max() <= 1           # Tanh
    assert D(img).shape == (8,)                          # un logit por imagen (sin sigmoide)


def test_conditional_models_accept_the_10_classes():
    G, D = CondGenerator(len(CLASSES)), CondDiscriminator(len(CLASSES))
    z = noise(0, 10)
    y = torch.arange(10)
    img = generate(G, z, y)
    assert img.shape == (10, 1, 28, 28) and img.abs().max() <= 1
    assert D(img, y).shape == (10,)
    # Con el mismo z, la clase cambia la imagen (la etiqueta sí entra a la red).
    same_z = z[:1].repeat(10, 1)
    out = generate(G, same_z, y)
    assert all(not torch.equal(out[0], out[i]) for i in range(1, 10))
    with pytest.raises(ValueError):
        generate(G, z)                                   # la cGAN exige etiquetas


def test_detach_keeps_the_discriminator_step_away_from_G():
    """Paso de D con fake.detach(): G no recibe gradientes. Paso de G: sí los recibe."""
    torch.manual_seed(0)
    G, D = Generator(), Discriminator()
    bce = nn.BCEWithLogitsLoss()
    fake = G(torch.randn(4, G.nz))
    loss_D = bce(D(torch.rand(4, 1, 28, 28) * 2 - 1), torch.ones(4)) + bce(D(fake.detach()), torch.zeros(4))
    loss_D.backward()
    assert all(p.grad is None for p in G.parameters())
    assert all(p.grad is not None for p in D.parameters())
    bce(D(fake), torch.ones(4)).backward()               # pérdida no saturante del generador
    assert all(p.grad is not None for p in G.parameters())


# ---------- determinismo y checkpoints ----------

def test_same_seed_same_images_and_batch_independence():
    torch.manual_seed(0)
    G = Generator()
    a = generate_from_seed(G, 42, 6)
    b = generate_from_seed(G, 42, 6)
    assert torch.equal(a, b)
    assert not torch.equal(a, generate_from_seed(G, 43, 6))
    z = noise(42, 6)
    # En modo eval BatchNorm usa estadísticas guardadas: una imagen no depende del resto del lote.
    assert torch.allclose(generate(G, z)[:2], generate(G, z[:2]), atol=1e-6)
    assert G.training                                    # generate() restaura el modo original


def test_noise_does_not_touch_global_rng():
    torch.manual_seed(1)
    expected = torch.rand(3)
    torch.manual_seed(1)
    noise(5, 4)
    assert torch.equal(torch.rand(3), expected)


def test_interpolation_endpoints():
    z0, z1 = noise(1, 1)[0], noise(2, 1)[0]
    path = interpolate(z0, z1, 10)
    assert path.shape == (10, z0.numel())
    assert torch.allclose(path[0], z0) and torch.allclose(path[-1], z1)
    assert torch.allclose(path[5] - path[4], path[1] - path[0], atol=1e-6)   # pasos iguales (lineal)


@pytest.mark.parametrize("conditional", [False, True])
def test_checkpoint_roundtrip(tmp_path, conditional):
    torch.manual_seed(0)
    G, D = (CondGenerator(10), CondDiscriminator(10)) if conditional else (Generator(), Discriminator())
    path = tmp_path / "g.pt"
    save_checkpoint(path, G, D, epochs=1, seed=42, classes=CLASSES)
    G2, ckpt = load_generator(path)
    assert ckpt["nz"] == G.nz and ckpt["base"] == G.base and ckpt["classes"] == CLASSES
    labels = torch.arange(5) if conditional else None
    assert torch.equal(generate_from_seed(G, 7, 5, labels), generate_from_seed(G2, 7, 5, labels))


@needs_models
def test_saved_checkpoints_load_and_generate():
    for path, conditional in ((GAN_CKPT, False), (CGAN_CKPT, True)):
        G, ckpt = load_generator(path)
        assert ckpt["conditional"] == conditional and ckpt["classes"] == CLASSES
        assert ckpt["seed"] == 42 and ckpt["epochs"] > 0
        img = generate_from_seed(G, 3, 4, torch.arange(4) if conditional else None)
        assert img.shape == (4, 1, 28, 28) and img.abs().max() <= 1


# ---------- métricas ----------

def test_fid_known_gaussians():
    # N(0, I) vs N(1, 4I) en 3 dimensiones: 3·1² + 3·(1 + 4 − 2·2) = 6
    assert fid((np.zeros(3), np.eye(3)), (np.ones(3), 4 * np.eye(3))) == pytest.approx(6.0)


def test_class_entropy_extremes():
    assert class_entropy(torch.arange(1000) % 10) == pytest.approx(1.0)
    assert class_entropy(torch.zeros(100, dtype=torch.long)) == pytest.approx(0.0)


def test_nearest_neighbor_and_diversity():
    pool = torch.randn(50, 1, 28, 28)
    d, idx = nearest_neighbors(pool[[7, 3]] + 1e-3, pool)
    assert idx.tolist() == [7, 3] and (d < 0.1).all()
    assert mean_pairwise_distance(pool[:1].repeat(10, 1, 1, 1)) == pytest.approx(0.0, abs=1e-5)


# ---------- póster y pantalla ----------

def test_poster_composes_without_window():
    from task_d_gan.poster import SIZE, make_poster
    pygame.font.init()
    poster = make_poster(torch.rand(10, 1, 28, 28) * 2 - 1, seed=7)
    assert poster.get_size() == SIZE


@needs_models
def test_screen_headless_flow(monkeypatch, tmp_path):
    """Abre la pantalla sin ventana, genera, cambia semilla, modelo y prenda, crea el póster y vuelve con ESC."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from common import ui
    from task_d_gan import poster, screen
    monkeypatch.setattr(poster, "POSTER_DIR", tmp_path)
    surface = ui.create_window((1024, 640), "test")
    view = screen.GanScreen(screen.load_cached(), size=surface.get_size())
    view.draw(surface)
    first = view.images.clone()
    click = lambda rect: view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))

    click(view.generate_button.rect)                         # botón Generar: semilla nueva, imágenes nuevas
    assert not torch.equal(first, view.images)
    seed, before = view.seed, view.images.clone()
    view.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))   # semilla + 1: otras imágenes
    assert view.seed == seed + 1 and not torch.equal(before, view.images)
    view.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))    # de vuelta: las mismas
    assert view.seed == seed and torch.equal(before, view.images)
    view.draw(surface)
    click(view.hit[("model", "cgan")])
    assert view.model == "cgan"
    view.draw(surface)
    click(view.hit[("class", 7)])                            # Tenis
    assert view.class_idx == 7
    view.draw(surface)
    click(view.hit[("view", "interp")])
    view.draw(surface)
    click(view.poster_button.rect)
    assert view.poster is not None and (tmp_path / f"poster_seed_{view.seed}.png").exists()
    view.draw(surface)
    view.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))   # cierra el póster
    assert view.poster is None and not view.done
    view.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))   # vuelve al menú
    assert view.done
    pygame.quit()


@needs_models
def test_screen_same_seed_same_images(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from common import ui
    from task_d_gan import screen
    surface = ui.create_window((1024, 640), "test")
    a = screen.GanScreen(screen.load_cached(), size=surface.get_size())
    b = screen.GanScreen(screen.load_cached(), size=surface.get_size())
    b.set_seed(500)
    b.set_seed(a.seed)
    assert torch.equal(a.images, b.images)
    pygame.quit()


@needs_models
def test_evaluation_summary_has_the_screen_numbers():
    if not EVAL_JSON.exists():
        pytest.skip("falta python -m task_d_gan.evaluate")
    import json
    s = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    assert 0 <= s["cgan"]["control_accuracy"] <= 1 and 0 <= s["dcgan"]["classes_covered"] <= 10
    assert set(s["cgan"]["control_by_class_pct"]) == set(CLASSES)


# ---------- robustez entre semillas ----------

def test_seed_study_summary_and_placement():
    """Media y desviación SIN la semilla 42; la 42 se ubica frente al rango de las otras 5 (no frente a sí misma)."""
    import pandas as pd
    from task_d_gan.seeds import METRICS, SEEDS, summarize, where_final_falls
    assert 42 not in SEEDS and len(SEEDS) == 5
    rows = []
    for kind, metrics in METRICS.items():
        for i, seed in enumerate([42] + SEEDS):
            # semilla 42 = 3,0 en todas las métricas; las otras, 1..5 (media 3, desv. est. 1,58)
            rows.append({"model": kind, "seed": seed, "final": seed == 42, "train_seconds": 1.0,
                         **{col: 3.0 if seed == 42 else float(i) for col, _, _ in metrics}})
    df = pd.DataFrame(rows)
    summary = summarize(df).set_index(["model", "seed"])
    assert summary.loc[("cgan", "media (5 semillas)"), "fid_cnn"] == pytest.approx(3.0)
    assert summary.loc[("cgan", "desv. est. (5 semillas)"), "fid_cnn"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    place = where_final_falls(df)
    assert place["in_range"].all() and (place["z_42"].abs() < 1e-9).all()
    fid_row = place[(place["model"] == "dcgan") & (place["metric"] == "FID-CNN")].iloc[0]
    assert fid_row["rank_42_of_6"] == 3           # menor es mejor: solo 1 y 2 son mejores que 3
