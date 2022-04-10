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

        self.nodes = None

    def update_nodes(self, activations):
        self.nodes = self.get_all_node_rects(activations)

    def draw_nodes(self):

        if self.nodes is not None:

            for node in self.nodes:
                rect = node['rect']
                color = node['color']
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 0, 0), rect, width=1)


    def get_all_node_rects(self, activations):
        C = len(activations)
        node_rects = []
        for c in range(1, C):
            number_nodes = len(activations[f'A{c}'].flatten())
            x_pos = int(self.topleft_surf[0] * 1.2) + (self.screen.get_width() // 2) // (C - 1) * (c - 1)
            w = 20
            h = self.screen.get_height() // number_nodes
            for n, a in enumerate(activations[f'A{c}'].flatten()):
                color = tuple([int(255 * a) for _ in range(3)])
                node_rects.append({'color': color, 'rect': (x_pos, n * h, w, h)})

        return node_rects


    def draw_background(self):
        self.screen.blit(self.surf, self.topleft_surf)

    def update(self):
        self.draw_background()
        self.draw_nodes()





