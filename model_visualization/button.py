import pygame


class Button:
    """Clickable button"""

    WHITE = (252, 252, 252)
    BLACK = (0, 0, 0)
    GREY = (200, 200, 200)
    GREY2 = (150, 150, 150)

    def __init__(self, x: int, y: int, text: str, size: int = 20):
        """

        :param x: topleft x position
        :param y: topleft y position
        :param text: button text
        :param size: text size
        """
        self.screen = pygame.display.get_surface()
        self.x = x
        self.y = y

        base_font = pygame.font.Font(None, int(size))
        self.text_surface = base_font.render(str(text), True, self.BLACK)
        self.width = self.text_surface.get_width() * 1.4
        self.height = self.text_surface.get_height() * 1.4
        self.bg_surface = pygame.Surface((self.width, self.height))
        self.bg_surface2 = pygame.Surface((self.width + 1, self.height + 1))
        self.bg_surface3 = pygame.Surface((self.width + 2, self.height + 2))
        self.rect = self.bg_surface.get_rect(topleft=(self.x, self.y))

        self.clicked = False

        self.history = []

    def draw(self) -> None:
        """Draw button"""
        if self.clicked:
            self.bg_surface.fill(self.GREY)
        else:
            self.bg_surface.fill(self.WHITE)
            self.bg_surface2.fill(self.GREY)
            self.bg_surface3.fill(self.GREY2)

        if not self.clicked:
            self.screen.blit(self.bg_surface3, self.rect)
            self.screen.blit(self.bg_surface2, self.rect)

        self.screen.blit(self.bg_surface, self.rect)
        self.screen.blit(self.text_surface, (self.rect[0] + self.width/2 - self.text_surface.get_width()/2,
                                             self.rect[1] + self.height/2 - self.text_surface.get_height()/2))

    def interact(self) -> None:
        """Interact with mouse clicking"""
        mouse_pos = pygame.mouse.get_pos()

        pressed = pygame.mouse.get_pressed()[0]

        self.rect.x = self.x
        self.rect.y = self.y

        self.clicked = False

        if pressed:
            if self.rect.collidepoint(*mouse_pos):
                self.clicked = True
                self.rect.x += 2
                self.rect.y += 2

    def check_released(self, must_be_in=True) -> bool:
        """Check if button is released"""
        if must_be_in:
            return self.history == [True, False] and self.rect.collidepoint(*pygame.mouse.get_pos())
        else:
            return self.history == [True, False]

    def update(self) -> None:
        self.interact()
        self.draw()

        self.history.append(self.clicked)

        if len(self.history) >= 2:
            self.history = self.history[-2:]
