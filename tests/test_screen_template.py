"""Prueba rápida de la plantilla de pantalla (sin ventana: driver de video dummy)."""
import pygame


def test_template_opens_draws_and_returns_with_esc(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from common import screen_template, ui

    screen = ui.create_window((1024, 640), "UrbanStyle")
    clock = pygame.time.Clock()
    view = screen_template.TemplateScreen(size=screen.get_size())
    assert screen.get_rect().contains(view.main) and screen.get_rect().contains(view.panel)

    view.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=view.main.center, rel=(0, 0), buttons=(0, 0, 0)))
    view.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=view.button.rect.center))
    assert view.clicks == 1
    view.update(1 / 60)
    view.draw(screen)  # tooltip y aviso incluidos

    # Contrato con el menú: run(screen, clock) vuelve con ESC.
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="", scancode=0))
    screen_template.run(screen, clock)
    pygame.quit()
