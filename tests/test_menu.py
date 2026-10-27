"""Pruebas del menú principal (sin ventana: driver de video dummy)."""
import pygame


def test_menu_has_four_tasks_and_draws(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import main
    pygame.init()
    surface = pygame.display.set_mode((1024, 640))
    cards, quit_button, static = main.build_menu(surface.get_size())
    assert [button.badge for button, _, _ in cards] == ["A", "B", "C", "D"]
    assert all(callable(module.run) for _, module, _ in cards)  # contrato run(screen, clock)
    rects = [button.rect for button, _, _ in cards] + [quit_button.rect]
    assert all(surface.get_rect().contains(r) for r in rects)  # nada se sale de la ventana
    assert not any(a.colliderect(b) for i, a in enumerate(rects) for b in rects[i + 1:])  # sin superposición
    # Lista tipográfica: "en desarrollo" como texto pequeño en las tareas que aún son marcadores.
    assert [button.subtitle.endswith("en desarrollo") for button, _, _ in cards] == [True, False, True, False]
    main.draw_menu(surface, cards, quit_button, static)
    pygame.quit()
