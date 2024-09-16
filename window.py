from typing import List
import numpy as np
import pygame
import sys
from random import randint, choice
from blur import blur
from particle_compute import compute_particle_pos
from screen import Screen

pygame.init()
screen = pygame.display.set_mode(size=Screen.window_size)
pygame.display.set_caption("Pygame Window")
surface = pygame.Surface(Screen.window_size)
surface.fill((0, 0, 0))

def gen_particles(n: int) -> List[List[int]]:
    return [
        [randint(0, Screen.windowx - 1), randint(0, Screen.windowy - 1), choice([-1, 1]), choice([-1, 1])]
        for i in range(0, n)
    ]

particles = np.array(gen_particles(100))
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    for i in range(0, 3):
        particles = compute_particle_pos(particles)

        # Create a temporary surface for particle drawing
        temp_surface = pygame.Surface(Screen.window_size, pygame.SRCALPHA)
        
        if i:
            # Blur the surface pixels
            surface_array = pygame.surfarray.array3d(surface)
            blurred_pixels = blur(surface_array)
            pygame.surfarray.blit_array(surface, blurred_pixels)

        # Draw particles on temporary surface
        for p in particles:
            x, y = int(round(p[0])), int(round(p[1]))
            temp_surface.set_at((x, y), (255, 255, 255)) # TODO fix this to update all particles as once, for some reason this is needlessly hard to do while still having blur

        # Blit temporary surface to main surface
        surface.blit(temp_surface, (0, 0))

        # Blit the blurred surface to the screen
        screen.blit(surface, (0, 0))
        pygame.display.update()