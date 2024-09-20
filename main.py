import os
import numpy as np
import sys
from random import randint, uniform

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True' # Suppress pygame welcome msg
import pygame   

from gpu.blur import blur
from gpu.particle_compute import compute_particle_pos
from settings import Settings

settings = Settings()

pygame.init()
if settings.FULLSCREEN:
    screen = pygame.display.set_mode(settings.WINDOW_SIZE, pygame.FULLSCREEN)
else:
    screen = pygame.display.set_mode(settings.WINDOW_SIZE, pygame.NOFRAME)

icon = pygame.Surface((32, 32))
icon.set_colorkey((0, 0, 0))
pygame.display.set_icon(icon)

pygame.display.set_caption("Slime Mold Simulation")
surface = pygame.Surface(settings.SURFACE_SIZE)
surface.fill((0, 0, 0))

def gen_particles(n):
    return [
    [
        randint(0, settings.SURFACEX - 1), # px
        randint(0, settings.SURFACEY - 1), # py
        uniform(-1, 1) * np.pi,            # theta
        0, # placeholder r
        0, # placeholder g
        0  # placeholder b
    ] for _ in range(n)]    

def set_particle_colours(particles):
    species = []

    if settings.RED[0]:   species.append(settings.RED[1])
    if settings.GREEN[0]: species.append(settings.GREEN[1])
    if settings.BLUE[0]:  species.append(settings.BLUE[1])

    # Lazy but effective fix for when random generation chooses zero species
    if len(species) == 0: species.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    
    particles_per_species = round(len(particles) / len(species)) 

    i = 0
    for s in species: 
        for _ in range(particles_per_species):
            particles[i] = [
                particles[i][0],
                particles[i][1],
                particles[i][2],
                s[0], # r
                s[1], # g
                s[2]  # b
            ]

            i += 1
            if i == len(particles):
                return particles

    return particles

particles = gen_particles(settings.AGENTS)
particles = np.array(set_particle_colours(particles))

i = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if i == settings.RANDOMIZATION_TIME:
        # Randomize settings after a specified period of time! Very cool
        settings = Settings()
        particles = set_particle_colours(particles)
        i = 0
    
    i += 1

    # Get current surface, compute new particle positions and place them on the surface
    surface_array = pygame.surfarray.array3d(surface)
    particles = compute_particle_pos(particles, surface_array, settings)

    ps = particles[:, :6].astype(int)
    rgb_values = np.stack((ps[:, 3], ps[:, 4], ps[:, 5]), axis=1)
    surface_array[ps[:, 0], ps[:, 1]] = rgb_values

    # Box blur screen so particle trails diffuse over time
    blurred_array = blur(surface_array, settings)

    pygame.surfarray.blit_array(surface, blurred_array)
    scaled_surface = pygame.transform.smoothscale(surface, settings.WINDOW_SIZE)
    screen.blit(scaled_surface, (0, 0))
    pygame.display.update()