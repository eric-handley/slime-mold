import math
from random import choice, randint, uniform

class Settings:
    AGENTS = randint(5000, 50000)

    RED = choice([True, False]), (randint(175, 255), randint(0, 125), randint(0, 125))
    GREEN = choice([True, False]), (randint(0, 125), randint(175, 255), randint(0, 125))
    BLUE = choice([True, False]), (randint(0, 125), randint(0, 125), randint(175, 255))

    DECAY_SPEED = randint(1, 10)

    SAMPLE_ANGLE = math.pi / randint(2, 9)

    SAMPLE_DISTANCE = randint(2, 35)
    SAMPLE_RADIUS = randint(2, 10)

    ATTRACT_WEIGHT = uniform(-1, 1)
    AVOID_WEIGHT = uniform(-1, 1)

    VELOCITY = uniform(0.1, 10)
    COHESION = uniform(0.7, 2)
    TURN_RANDOMNESS = uniform(0, 0.5)

    TURN_WEIGHT_LEFT = (-1 * SAMPLE_ANGLE * COHESION)
    TURN_WEIGHT_RIGHT = (SAMPLE_ANGLE * COHESION)

    FULLSCREEN = False
    WINDOWX = 1200
    WINDOWY = 800
    WINDOW_SIZE = (WINDOWX, WINDOWY)

    SURFACEX = round(WINDOWX * 1)
    SURFACEY = round(WINDOWY * 1)
    SURFACE_SIZE = (SURFACEX, SURFACEY)


# Sample default settings if you'd like to tweak manually :)
# class Settings: 
#     AGENTS = 50000

#     RED = True, (175, 0, 125)
#     GREEN = True, (50, 175, 100)
#     BLUE = True, (130, 75, 175)

#     DECAY_SPEED = 15

#     SAMPLE_ANGLE = math.pi / 8

#     SAMPLE_DISTANCE = 5
#     SAMPLE_RADIUS = 5

#     ATTRACT_WEIGHT = 0.3
#     AVOID_WEIGHT = -0.1

#     VELOCITY = 4
#     COHESION = 0.8
#     TURN_RANDOMNESS = 0.2

#     TURN_WEIGHT_LEFT = (-1 * SAMPLE_ANGLE * COHESION)
#     TURN_WEIGHT_RIGHT = (SAMPLE_ANGLE * COHESION)

#     FULLSCREEN = False
#     WINDOWX = 1200
#     WINDOWY = 800
#     WINDOW_SIZE = (WINDOWX, WINDOWY)

#     SURFACEX = round(WINDOWX * 0.5)
#     SURFACEY = round(WINDOWY * 0.5)
#     SURFACE_SIZE = (SURFACEX, SURFACEY)