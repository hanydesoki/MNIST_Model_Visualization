import pygame

from neural_network import NeuralNetwork


class ModelVisualization:
    """class that show in real time neuron interaction when we make a prediction"""

    BACKGROUND_COLOR = (0, 0, 50)

    def __init__(self, model: NeuralNetwork):
        self.model = model
        self.screen = pygame.display.get_surface()
        self.topleft_surf = (self.screen.get_width() // 2, 0)
        self.surf = pygame.Surface((self.screen.get_width() // 2, self.screen.get_height()))
        self.surf.fill(self.BACKGROUND_COLOR)

        self.font = pygame.font.SysFont('comicsans', size=10)

        self.nodes = None

    def update_nodes(self, activations) -> None:
        """Update nodes"""
        self.nodes = self.get_all_node_rects(activations)

    def draw_nodes(self) -> None:
        """Draw all nodes for each layers with rectangles"""
        if self.nodes is not None:

            for node in self.nodes:
                rect = node['rect']
                color = node['color']
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 0, 0), rect, width=1)


    def get_all_node_rects(self, activations) -> list:
        """return all nodes from activations in a list.
        Each node is a dictionary that contain color, activation value and its rect"""
        C = len(activations)
        node_rects = []
        for c in range(1, C):
            number_nodes = len(activations[f'A{c}'].flatten())
            x_pos = int(self.topleft_surf[0] * 1.2) + (self.screen.get_width() // 2) // (C - 1) * (c - 1)
            w = 20
            h = self.screen.get_height() // number_nodes
            for n, a in enumerate(activations[f'A{c}'].flatten()):
                color = tuple([int(255 * a) for _ in range(3)])
                node_rects.append({'color': color, 'rect': (x_pos, n * h, w, h), 'val': a})

        return node_rects

    def activation_interact(self) -> None:
        """Highlight neuron and show activation value when hovering"""
        for activation in self.nodes:
            rect = pygame.Rect(activation['rect'])
            val = activation['val']

            if rect.collidepoint(*pygame.mouse.get_pos()):
                text_surf = self.font.render(str(val), True, 'white')
                x = rect.right + 3
                y = rect.centery
                pygame.draw.rect(self.screen, 'red', rect, width=1)
                self.screen.blit(text_surf, (x, y))

    def draw_background(self) -> None:
        self.screen.blit(self.surf, self.topleft_surf)

    def update(self) -> None:
        self.draw_background()
        self.draw_nodes()
        self.activation_interact()





