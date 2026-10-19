"""Menú principal de UrbanStyle (Pygame)."""
import sys

import pygame

from common import config
from common.ui import Button
from task_a_pathfinding import screen as screen_a
from task_b_forecasting import screen as screen_b
from task_c_pricing import screen as screen_c
from task_d_gan import screen as screen_d

TASKS = [
    ("Tarea A", screen_a),
    ("Tarea B", screen_b),
    ("Tarea C", screen_c),
    ("Tarea D", screen_d),
]


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("UrbanStyle")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont(None, 64)

    bw, bh, gap = 320, 56, 18
    x = (config.WINDOW_WIDTH - bw) // 2
    y0 = 180
    buttons = [
        (Button((x, y0 + i * (bh + gap), bw, bh), label), module)
        for i, (label, module) in enumerate(TASKS)
    ]
    quit_button = Button((x, y0 + len(TASKS) * (bh + gap), bw, bh), "Salir")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or quit_button.handle_event(event):
                running = False
                break
            for button, module in buttons:
                if button.handle_event(event):
                    module.run(screen, clock)
                    break

        screen.fill(config.COLOR_BG)
        title = title_font.render("UrbanStyle", True, config.COLOR_TEXT)
        screen.blit(title, title.get_rect(center=(config.WINDOW_WIDTH // 2, 100)))
        for button, _ in buttons:
            button.draw(screen)
        quit_button.draw(screen)
        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
