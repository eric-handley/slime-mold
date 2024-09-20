import os
import numpy as np
import sys
from random import randint, uniform

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True' # Suppress pygame welcome msg
import pygame   

from gpu.blur import blur
from gpu.particle_compute import compute_particle_pos
from settings import Settings

pygame.init()
if Settings.FULLSCREEN:
    screen = pygame.display.set_mode(Settings.WINDOW_SIZE, pygame.FULLSCREEN)
else:
    screen = pygame.display.set_mode(Settings.WINDOW_SIZE, pygame.NOFRAME)

icon = pygame.Surface((32, 32))
icon.set_colorkey((0, 0, 0))
pygame.display.set_icon(icon)

pygame.display.set_caption("Slime Mold Simulation")
surface = pygame.Surface(Settings.SURFACE_SIZE)
surface.fill((0, 0, 0))

def gen_particles(n):
    species = []

    if Settings.RED[0]:   species.append(Settings.RED[1])
    if Settings.GREEN[0]: species.append(Settings.GREEN[1])
    if Settings.BLUE[0]:  species.append(Settings.BLUE[1])

    # Lazy but effective fix for when random generation chooses zero species
    if len(species) == 0: species.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    particles_per_species = round(n / len(species)) 
    particles = []
    
    for s in species: 
        particles += [
            [
                randint(0, Settings.SURFACEX - 1), # px
                randint(0, Settings.SURFACEY - 1), # py
                uniform(-1, 1) * np.pi,            # theta
                s[0], # r
                s[1], # g
                s[2]  # b
            ] for _ in range(0, particles_per_species)
        ]

    return particles

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
    scaled_surface = pygame.transform.smoothscale(surface, Settings.WINDOW_SIZE)
    screen.blit(scaled_surface, (0, 0))
    pygame.display.update()