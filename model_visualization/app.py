import pygame

from .model_visualization import ModelVisualization
from .button import Button
from .drawboard import DrawBoard

import sys

class App:
    """class that manage interaction between
            the model visualization and the drawboard"""

    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 650

    BACKGROUND_COLOR = (50, 50, 50)

    def __init__(self, model):
        """

        :param model: trained NeuralNetwork model
        """

        self.model = model

        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("MNIST Visualization Model")

        self.background = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.background.fill(self.BACKGROUND_COLOR)

        self.drawboard = DrawBoard()
        self.model_visualization = ModelVisualization(model)

        self.activations = None
        self.predicted_value = None

        self.font = pygame.font.SysFont('comicsans', size=30)

        self.clear_button = Button(20, 20, 'Clear board', size=20)

        self.reset_board()


    def draw_background(self):
        self.screen.blit(self.background, (0, 0))

    def draw_midline(self):
        pygame.draw.line(self.screen, 'white',
                         start_pos=(self.SCREEN_WIDTH // 2, 0),
                         end_pos=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT),
                         width=2)

    def interact(self, all_events):
        """Manage to clear board with 'c' or with the 'Clear board' button"""
        for event in all_events:
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_c:
                    self.reset_board()

        if self.clear_button.check_released():
            self.reset_board()


    def reset_board(self):
        """Clear board"""
        self.drawboard.reset_board()
        self.predict()

    def update_activations(self, X):
        """Update actications attribute by getting activations dictionary with forward propagation"""
        self.activations = self.model.forward_propagation(X, self.model.params)

    def predict(self):
        """Get numpy array from the drawboard and predict digit with the model. Update predicted value attribute"""
        X = self.drawboard.get_array().T.T.T.reshape(1, DrawBoard.N_PIXEL_X * DrawBoard.N_PIXEL_Y)

        self.predicted_value = self.model.predict(X.T)[:,0].argmax()

        #print(predicted_value)
        self.update_activations(X.T)
        self.model_visualization.update_nodes(self.activations)

    def show_prediction(self):
        """Show the predicted value on top of the drawboard"""
        if self.predicted_value is not None:

            text_surf = self.font.render(f'Prediction: {self.predicted_value}', True, 'orange')

            rect = text_surf.get_rect(center=(self.screen.get_width() // 4,
                                              self.drawboard.rect.top - self.drawboard.topleft_surf[1] // 2))

            self.screen.blit(text_surf, rect)

    def run(self):
        """Run application"""
        while True:
            all_events = pygame.event.get()
            for event in all_events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.draw_background()

            if sum(pygame.mouse.get_pressed()) and self.drawboard.rect.collidepoint(*pygame.mouse.get_pos()):
                self.predict()

            self.interact(all_events)
            self.clear_button.update()
            self.drawboard.update()
            self.show_prediction()
            self.model_visualization.update()

            self.draw_midline()

            pygame.display.update()