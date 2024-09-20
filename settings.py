import math
from random import choice, randint, uniform

class Settings:
    def __init__(self):
        self.AGENTS = 25000

        self.FULLSCREEN = False
        self.WINDOWX = 1200
        self.WINDOWY = 800
        self.WINDOW_SIZE = (self.WINDOWX, self.WINDOWY)

        self.SURFACEX = round(self.WINDOWX * 1) # Up/downscale factor
        self.SURFACEY = round(self.WINDOWY * 1) # Up/downscale factor
        self.SURFACE_SIZE = (self.SURFACEX, self.SURFACEY)
    
        self.RANDOMIZATION_TIME = 130

        self.randomize()
    
    def randomize(self):
        # If you want to manually set constant settings, just set parameters to constants
        self.RED = choice([True, False]), (randint(175, 255), randint(0, 125), randint(0, 125))
        self.GREEN = choice([True, False]), (randint(0, 125), randint(175, 255), randint(0, 125))
        self.BLUE = choice([True, False]), (randint(0, 125), randint(0, 125), randint(175, 255))

        self.DECAY_SPEED = randint(1, 10)

        self.SAMPLE_ANGLE = math.pi / randint(2, 9)

        self.SAMPLE_DISTANCE = randint(2, 35)
        self.SAMPLE_RADIUS = randint(2, 10)

        self.ATTRACT_WEIGHT = uniform(-1, 1)
        self.AVOID_WEIGHT = uniform(-1, 1)

        self.VELOCITY = uniform(0.1, 10)
        self.COHESION = uniform(0.7, 2)
        self.TURN_RANDOMNESS = uniform(0, 0.5)

        self.TURN_WEIGHT_LEFT = (-1 * self.SAMPLE_ANGLE * self.COHESION)
        self.TURN_WEIGHT_RIGHT = (self.SAMPLE_ANGLE * self.COHESION)