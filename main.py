import numpy as np
import pygame
import sys
from random import randint, uniform

from gpu.blur import blur
from gpu.particle_compute import compute_particle_pos
from settings import Settings

pygame.init()
if Settings.FULLSCREEN:
    screen = pygame.display.set_mode(Settings.WINDOW_SIZE, pygame.FULLSCREEN)
else:
    screen = pygame.display.set_mode(Settings.WINDOW_SIZE)

pygame.display.set_caption("Window")
surface = pygame.Surface(Settings.WINDOW_SIZE)
surface.fill((0, 0, 0))

def gen_particles(n):
    return [
        [
            randint(0, Settings.WINDOWX - 1), # px
            randint(0, Settings.WINDOWY - 1), # py
            uniform(-1, 1) * np.pi,           # theta
            (255 if i % 3 == 0 and Settings.RED   else 0), # r
            (255 if i % 3 == 1 and Settings.GREEN else 0), # g
            (255 if i % 3 == 2 and Settings.BLUE  else 0)  # b
        ] for i in range(0, n)
    ]

particles = np.array(gen_particles(Settings.AGENTS))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get current surface, compute new particle positions and place them on the surface
    surface_array = pygame.surfarray.array3d(surface)
    particles = compute_particle_pos(particles, surface_array)

    ps = particles[:, :6].astype(int)
    rgb_values = np.stack((ps[:, 3], ps[:, 4], ps[:, 5]), axis=1)
    surface_array[ps[:, 0], ps[:, 1]] = rgb_values

    # Box blur screen so particle trails diffuse over time
    blurred_array = blur(surface_array)

    pygame.surfarray.blit_array(surface, blurred_array)
    screen.blit(surface, (0, 0))
    pygame.display.update()