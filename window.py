import pygame
import sys
from random import randint
import numpy as np
from blur import blur

pygame.init()
windowx = 1500
windowy = 800
window_size = (windowx, windowy)
screen = pygame.display.set_mode(size=window_size)
pygame.display.set_caption("Pygame Window")
surface = pygame.Surface(window_size)
surface.fill((0, 0, 0))
screen.blit(surface, (0, 0))

def blur_pixels():
    pxs = pygame.surfarray.array3d(surface)
    blurred_pxs = blur(pxs)
    pygame.surfarray.blit_array(surface, blurred_pxs)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    blur_pixels()

    # Create random circles for testing
    for i in range(1, 2):
        r, g, b = randint(0, 255), randint(0, 255), randint(0, 255)
        color = (r << 16) + (g << 8) + b
        center = (randint(0, windowx - 1), randint(0, windowy - 1))
        radius = randint(1, 50)

        pygame.draw.circle(surface, color, center, radius)

    screen.blit(surface, (0, 0))
    pygame.display.flip()