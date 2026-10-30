"""Menú principal de UrbanStyle (Pygame): portada editorial que abre cada tarea con run(screen, clock)."""
import sys

import pygame

from common import config, ui
from common.ui import THEME, px
from task_a_pathfinding import screen as screen_a
from task_b_forecasting import screen as screen_b
from task_c_pricing import screen as screen_c
from task_d_gan import screen as screen_d

# (letra, nombre, descripción, módulo con run(screen, clock), ¿pantalla terminada?)
TASKS = [
    ("A", "Tarea A", "Rutas de clientes y layout", screen_a, True),
    ("B", "Tarea B", "Pronóstico de demanda", screen_b, True),
    ("C", "Tarea C", "Precios dinámicos con RL", screen_c, True),
    ("D", "Tarea D", "Imágenes promocionales con GAN", screen_d, True),
]
KEYS = {pygame.K_a: 0, pygame.K_b: 1, pygame.K_c: 2, pygame.K_d: 3}  # atajos de teclado


def build_menu(size):
    """Índice de las 4 tareas (lista tipográfica), enlace Salir y la capa estática (portada y pie).

    `size` es el tamaño físico de la ventana; la geometría se define en px lógicos y se escala con ui.px().
    """
    w, h = size[0] / THEME.scale, size[1] / THEME.scale  # tamaño lógico
    m = THEME.margin * 2
    list_x, list_w = w * 0.52, w * 0.48 - m
    row_h, top = 84, 168
    cards = []
    for i, (letter, name, desc, module, ready) in enumerate(TASKS):
        subtitle = name if ready else f"{name} · en desarrollo"
        rect = pygame.Rect(px(list_x), px(top + i * row_h), px(list_w), px(row_h))
        cards.append((ui.Button(rect, desc, subtitle=subtitle, badge=letter), module, ready))
    quit_w = ui.font(THEME.size_body, weight="medium").size("Salir")[0] / THEME.scale + 12
    quit_button = ui.Button((px(list_x), px(top + 4 * row_h + 22), px(quit_w), px(30)), "Salir")

    static = ui.background(size).copy()
    # Portada: etiqueta, nombre de la marca en serif y subtítulo.
    ui.blit_eyebrow(static, "Proyecto final", (px(m), px(170)), THEME.accent_text)
    title = ui.blit_text(static, "UrbanStyle", (px(m - 4), px(286)), THEME.size_hero, weight="serif",
                         anchor="bottomleft")
    pygame.draw.rect(static, THEME.accent, (px(m), title.bottom + px(10), px(56), max(2, px(3))))
    ui.blit_text(static, "Sistema de IA para retail de moda", (px(m), title.bottom + px(30)), 19,
                 THEME.text_muted, weight="medium")
    # Índice: etiqueta y líneas finas entre filas.
    ui.blit_eyebrow(static, "Contenido", (px(list_x), px(top - 30)))
    for i in range(len(TASKS) + 1):
        ui.hairline(static, px(list_x), px(list_x + list_w), px(top + i * row_h))
    ui.blit_text(static, "Clic o teclas A, B, C, D", (px(list_x + list_w), px(top + 4 * row_h + 26)),
                 THEME.size_caption, THEME.text_faint, anchor="topright")
    # Pie.
    ui.hairline(static, px(m), size[0] - px(m), size[1] - px(48))
    ui.blit_text(static, "Elizabeth Correa · Sebastián Ortega", (px(m), size[1] - px(28)), THEME.size_label,
                 THEME.text_faint, anchor="midleft")
    return cards, quit_button, static


def draw_menu(screen, cards, quit_button, static) -> None:
    screen.blit(static, (0, 0))
    for button, _, _ in cards:
        button.draw(screen)
    quit_button.draw(screen)


def main() -> None:
    screen = ui.create_window((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), "UrbanStyle")
    clock = pygame.time.Clock()
    cards, quit_button, static = build_menu(screen.get_size())
    fade = ui.FadeIn()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or quit_button.handle_event(event):
                running = False
                break
            chosen = KEYS.get(event.key) if event.type == pygame.KEYDOWN else None
            for i, (button, module, _) in enumerate(cards):
                if button.handle_event(event) or chosen == i:
                    module.run(screen, clock)
                    fade = ui.FadeIn()  # aparición suave al volver al menú
                    break

        draw_menu(screen, cards, quit_button, static)
        fade.draw(screen)
        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
