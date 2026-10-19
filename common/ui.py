"""Componentes de interfaz Pygame: Button y Slider (responsable: Elizabeth)."""
import pygame

from common import config


class Button:
    def __init__(self, rect, label: str, font: pygame.font.Font | None = None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font or pygame.font.SysFont(None, 32)

    def draw(self, surface: pygame.Surface) -> None:
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        color = config.COLOR_BUTTON_HOVER if hover else config.COLOR_BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        text = self.font.render(self.label, True, config.COLOR_TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """True si se hizo clic izquierdo sobre el botón."""
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


class Slider:
    def __init__(self, rect, min_value: float, max_value: float, value: float | None = None):
        self.rect = pygame.Rect(rect)
        self.min_value = min_value
        self.max_value = max_value
        self.value = min_value if value is None else value
        self.dragging = False

    def _set_from_x(self, x: int) -> None:
        t = (x - self.rect.left) / max(1, self.rect.width)
        t = min(1.0, max(0.0, t))
        self.value = self.min_value + t * (self.max_value - self.min_value)

    def handle_event(self, event: pygame.event.Event) -> float:
        """Procesa clic/arrastre y devuelve el valor actual."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.inflate(0, 20).collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
        return self.value

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, config.COLOR_BUTTON, self.rect, border_radius=4)
        span = (self.max_value - self.min_value) or 1
        t = (self.value - self.min_value) / span
        x = self.rect.left + int(t * self.rect.width)
        pygame.draw.circle(surface, config.COLOR_ACCENT, (x, self.rect.centery), 10)
