"""Tarea D: pantalla Pygame (responsable: Elizabeth)."""
import pygame

from common import config


def run(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    font = pygame.font.SysFont(None, 48)
    text = font.render("Tarea D – pendiente", True, config.COLOR_TEXT)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        screen.fill(config.COLOR_BG)
        screen.blit(text, text.get_rect(center=screen.get_rect().center))
        pygame.display.flip()
        clock.tick(config.FPS)
