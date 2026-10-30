"""Escala de la ventana en distintos sistemas y pantallas (sin abrir ventanas: choose_scale es una función pura)."""
import pytest

from common import config
from common.ui import choose_scale

BASE = (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

# Área útil en píxeles físicos (sin barra de tareas / menús) y escala del sistema.
SCREENS = {
    "Windows 1366×768 al 100 %": ((1366, 728), 1.0),
    "Windows 1920×1080 al 100 %": ((1920, 1032), 1.0),
    "Windows 1920×1080 al 125 %": ((1920, 1020), 1.25),
    "Windows 1920×1080 al 150 %": ((1920, 1008), 1.5),
    "Windows 2560×1440 al 150 %": ((2560, 1392), 1.5),
    "Windows 2560×1600 al 175 %": ((2560, 1544), 1.75),
    "Windows 3840×2160 al 200 %": ((3840, 2096), 2.0),
    "Mac 1470×956 pt": ((1470, 956 - 80), 1.0),
    "Mac 1280×800 pt": ((1280, 800 - 80), 1.0),
    "Linux 1920×1080": ((1920, 1080), 1.0),
    "Linux 1366×768": ((1366, 768), 1.0),
}


def original_scale(work_area, system_scale) -> float:
    """El algoritmo de antes del zoom (barra de título calculada una sola vez), para comparar."""
    scale, (work_w, work_h) = system_scale, work_area
    title_bar = 40 * scale
    w, h = BASE
    while scale > 1.0 and (w * scale > work_w - 16 or h * scale + title_bar > work_h - 8):
        scale = round(scale - 0.05, 2)
    return max(1.0, scale)


def fits(scale, work_area) -> bool:
    return BASE[0] * scale <= work_area[0] - 16 and (BASE[1] + 40) * scale <= work_area[1] - 8


@pytest.mark.parametrize("name", SCREENS)
def test_zoom_never_makes_the_window_smaller_and_always_fits(name):
    work, system = SCREENS[name]
    before = original_scale(work, system)
    now = choose_scale(BASE, work, system, config.UI_ZOOM)
    assert now >= before                                  # igual o más grande que antes
    assert now <= system * config.UI_ZOOM                 # nunca más que lo pedido
    assert now == 1.0 or fits(now, work)                  # cabe en la pantalla (1,0 es el mínimo de siempre)


@pytest.mark.parametrize("name, expected", [
    ("Mac 1470×956 pt", 1.25),                 # 1280×800: lo que se pidió
    ("Windows 1920×1080 al 150 %", 1.45),      # igual que antes: 1,5 no cabe con la barra de tareas de 72 px
    ("Windows 1920×1080 al 125 %", 1.45),
    ("Windows 1920×1080 al 100 %", 1.25),
    ("Windows 3840×2160 al 200 %", 2.5),
    ("Windows 1366×768 al 100 %", 1.05),       # cabe un poco más grande que antes (1,0)
])
def test_expected_scale_on_common_screens(name, expected):
    work, system = SCREENS[name]
    assert choose_scale(BASE, work, system, config.UI_ZOOM) == pytest.approx(expected)


def test_system_scale_is_kept_when_it_fits():
    """Con zoom que no cabe, se baja hasta la escala del sistema exacta, no a una escala vecina más pequeña."""
    work = (1920, 1032)                        # 1920×1080 al 150 % con la barra de tareas oculta o pequeña
    assert choose_scale(BASE, work, 1.5, 1.25) == pytest.approx(1.5)
    assert original_scale(work, 1.5) == pytest.approx(1.5)


def test_without_zoom_matches_old_behaviour_or_better():
    for work, system in SCREENS.values():
        assert choose_scale(BASE, work, system, 1.0) >= original_scale(work, system)
