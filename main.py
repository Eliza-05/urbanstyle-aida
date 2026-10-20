"""Menú principal de UrbanStyle (Pygame): abre la pantalla de cada tarea con run(screen, clock)."""
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
    ("A", "Tarea A", "Rutas de clientes y layout", screen_a, False),
    ("B", "Tarea B", "Pronóstico de demanda", screen_b, True),
    ("C", "Tarea C", "Precios dinámicos con RL", screen_c, False),
    ("D", "Tarea D", "Imágenes promocionales con GAN", screen_d, False),
]
KEYS = {pygame.K_a: 0, pygame.K_b: 1, pygame.K_c: 2, pygame.K_d: 3}  # atajos de teclado


def build_menu(size):
    """Tarjetas de las 4 tareas (cuadrícula 2×2), botón Salir y la capa estática (fondo, títulos, pie).

    `size` es el tamaño físico de la ventana; la geometría se define en px lógicos y se escala con ui.px().
    """
    w, h = size[0] / THEME.scale, size[1] / THEME.scale  # tamaño lógico
    card_w, card_h, gap = min(400, (w - 3 * 24) / 2), 96, 18
    x0 = (w - 2 * card_w - gap) / 2
    y0 = 238
    cards = []
    for i, (letter, name, desc, module, ready) in enumerate(TASKS):
        rect = pygame.Rect(px(x0 + (i % 2) * (card_w + gap)), px(y0 + (i // 2) * (card_h + gap)), px(card_w), px(card_h))
        cards.append((ui.Button(rect, name, subtitle=desc, badge=letter), module, ready))
    quit_button = ui.Button((px((w - 180) / 2), px(y0 + 2 * (card_h + gap) + 16), px(180), px(42)), "Salir")

    static = ui.background(size).copy()
    cx = size[0] // 2
    title = ui.blit_text(static, "UrbanStyle", (cx, px(118)), THEME.size_hero, weight="bold", anchor="center")
    line = ui.gradient_surface((px(96), px(4)), THEME.accent, THEME.accent_2, px(2))
    static.blit(line, line.get_rect(midtop=(cx, title.bottom + px(8))))
    ui.blit_text(static, "Sistema de IA para retail de moda", (cx, title.bottom + px(36)), 19, THEME.text_muted,
                 weight="medium", anchor="center")
    ui.blit_text(static, "Elige una tarea  ·  clic o teclas A, B, C, D", (cx, px(y0 - 22)), THEME.size_caption,
                 THEME.text_faint, anchor="center")
    footer_y = size[1] - px(30)
    half = (size[0] - px(48)) // 2  # divisor que se desvanece hacia ambos lados
    for i, (start, end) in enumerate(((0, 110), (110, 0))):
        piece = ui.gradient_surface((half, max(1, px(1))), THEME.accent, THEME.accent, alpha=(start, end))
        static.blit(piece, (px(24) + i * half, footer_y - px(18)))
    ui.blit_text(static, "Integrantes: Elizabeth Correa · Sebastián Ortega", (cx, footer_y), THEME.size_label,
                 THEME.text_muted, anchor="center")
    return cards, quit_button, static


def draw_menu(screen, cards, quit_button, static) -> None:
    screen.blit(static, (0, 0))
    for button, _, ready in cards:
        button.draw(screen)
        if not ready:  # etiqueta honesta: la pantalla todavía es un marcador de posición
            r = button.draw_rect
            ui.pill(screen, "en desarrollo", (r.right - px(14), r.top + px(12)), anchor="topright")
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
