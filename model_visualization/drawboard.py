import pygame

import numpy as np


class DrawBoard:
    """class for drawing numbers in a board and fill
    numpy array"""

    N_PIXEL_X = 28
    N_PIXEL_Y = 28

    SURF_OFFSET = 0.8

    def __init__(self):
        self.screen = pygame.display.get_surface()

        self.surf_width = int((self.screen.get_width() // 2) * self.SURF_OFFSET)
        self.surf_height = int((self.screen.get_height()) * self.SURF_OFFSET)

        self.topleft_surf = (((self.screen.get_width() // 2) - self.surf_width) // 2,
                             (self.screen.get_height() // 2) - self.surf_height // 2)

        self.surf = pygame.Surface((self.surf_width, self.surf_height))

        self.rect = self.surf.get_rect(topleft=self.topleft_surf)

        self.tile_size_x = self.surf_width // self.N_PIXEL_X
        self.tile_size_y = self.surf_height // self.N_PIXEL_Y

        self.reset_board()

    def reset_board(self):
        self.board = np.zeros((28, 28))

    def draw_surf(self):
        self.screen.blit(self.surf, self.rect)

    def draw_board(self):
        self.draw_surf()

        for i in range(self.N_PIXEL_X):
            for j in range(self.N_PIXEL_Y):
                if self.board[i, j]:
                    x = self.topleft_surf[0] + i * self.tile_size_x
                    y = self.topleft_surf[1] + j * self.tile_size_y
                    pygame.draw.rect(self.screen, 'white', (x, y, self.tile_size_x, self.tile_size_y))

    def interact(self):
        mouse_pos = pygame.mouse.get_pos()

        pressed_left = pygame.mouse.get_pressed()[0]

        pressed_right = pygame.mouse.get_pressed()[2]

        if self.rect.collidepoint(*mouse_pos) and sum([pressed_left, pressed_right]):
            i, j = self.get_array_indexes_from_pos(mouse_pos)

            if pressed_left: val = 1
            else: val = 0

            for inci, incj in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:

                try:
                    index_i = i + inci
                    index_j = j + incj

                    if index_i == - 1 or index_j == -1:
                        continue

                    self.board[index_i, index_j] = val
                except IndexError:
                    pass



    def get_array_indexes_from_pos(self, mouse_pos):
        mx, my = mouse_pos

        i = (mx - self.topleft_surf[0]) // self.tile_size_x
        j = (my - self.topleft_surf[1]) // self.tile_size_y

        return i, j

    def get_array(self):
        return self.board

    def update(self):
        self.interact()
        self.draw_board()