import math
import numpy as np
from numba import cuda
from numba.cuda import random
from settings import Settings


# Generate a list of pixel offsets to sample
# all pixels in a radius around a given x, y
def generate_pixel_offsets(radius):
    offsets = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if x**2 + y**2 <= radius**2:
                offsets.append((x, y))
    return offsets

# CUDA functions can't access Settings directly
SURFACEX = Settings.SURFACEX
SURFACEY = Settings.SURFACEY
SAMP_ANGLE = Settings.SAMPLE_ANGLE
SAMP_POS_LEFT = -1 * SAMP_ANGLE
SAMP_POS_RIGHT = SAMP_ANGLE
SAMP_DIST = Settings.SAMPLE_DISTANCE
SAMP_RAD = Settings.SAMPLE_RADIUS
SAMP_OFFSETS = generate_pixel_offsets(SAMP_RAD)
VELOCITY = Settings.VELOCITY
COHESION = Settings.COHESION
TURN_WEIGHT_LEFT = Settings.TURN_WEIGHT_LEFT
TURN_WEIGHT_RIGHT = Settings.TURN_WEIGHT_RIGHT
TURN_RANDOMNESS = Settings.TURN_RANDOMNESS
AVOID_WEIGHT = Settings.AVOID_WEIGHT
ATTRACT_WEIGHT = Settings.ATTRACT_WEIGHT

@cuda.jit
def sum_sample_pixels(sx, sy, screen, offsets, rgb):
    sum = 0
    r, g, b = rgb
    c = max(r, g, b)

    for i in range(offsets.shape[0]):
        x, y = offsets[i]
        spx = sx + x
        spy = sy + y

        if spx > 0 and spy > 0 and spx < SURFACEX and spy < SURFACEY:
            sr, sg, sb = screen[int(spx), int(spy)]
            sc = max(sr, sg, sb)
            
            # If max channel of sampled pixel matches max channel of particle
            if (sc == sr and c == r) or (sc == sb and c == b) or (sc == sg and c == g):
                sum += ATTRACT_WEIGHT
            if (sc == sr and c != r) or (sc == sb and c != b) or (sc == sg and c != g):
                sum += AVOID_WEIGHT
    
    return sum

@cuda.jit
def update_theta(p, screen, offsets, random_theta):
    px, py, theta, r, g, b = p

    # Sample three points in front of particle
    sx1 = px + math.cos(theta + SAMP_POS_LEFT) * SAMP_DIST
    sy1 = py + math.sin(theta + SAMP_POS_LEFT) * SAMP_DIST

    sx2 = px + math.cos(theta) * SAMP_DIST
    sy2 = py + math.sin(theta) * SAMP_DIST

    sx3 = px + math.cos(theta + SAMP_POS_RIGHT) * SAMP_DIST
    sy3 = py + math.sin(theta + SAMP_POS_RIGHT) * SAMP_DIST

    s1 = sum_sample_pixels(sx1, sy1, screen, offsets, (r,g,b))
    s2 = sum_sample_pixels(sx2, sy2, screen, offsets, (r,g,b))
    s3 = sum_sample_pixels(sx3, sy3, screen, offsets, (r,g,b))

    # # Find highest weighted point
    smax = max(s1, s2, s3)

    a, b = -TURN_RANDOMNESS, TURN_RANDOMNESS
    theta += a + (b - a) * random_theta

    # Check for tie, otherwise we will bias turning left when samples are equal
    if (smax == s1 and smax == s2) or (smax == s2 and smax == s3) or (smax == s1 and smax == s2):
        return theta

    theta_left = theta + TURN_WEIGHT_LEFT
    theta_right = theta + TURN_WEIGHT_RIGHT   

    # Turn toward highest sampled area
    if s1 == smax:
        return theta_left
    if s3 == smax:
        return theta_right
    
    # # Don't go straight if weight for s2 < 0
    # if s2 < 0:
    #     if s1 > s3:
    #         return theta_left
    #     else:
    #         return theta_right
        
    return theta


@cuda.jit
def update_pos(p):
    px, py, theta = p[0], p[1], p[2]

    # Find velocity based on current theta, update position by that velocity
    pvx = math.cos(theta) * VELOCITY
    pvy = math.sin(theta) * VELOCITY
    
    nx = px + pvx
    ny = py + pvy
    
    bounced = False
    
    # Particle "bounces" if it reaches edge of screen
    if nx < 1:
        px = 0
        pvx = abs(pvx)
        bounced = True
    elif nx >= SURFACEX:
        px = SURFACEX - 1
        pvx = -abs(pvx)
        bounced = True
    else:
        px = nx
    
    if ny < 1:
        py = 0
        pvy = abs(pvy)
        bounced = True
    elif ny >= SURFACEY:
        py = SURFACEY - 1
        pvy = -abs(pvy)
        bounced = True
    else:
        py = ny
    
    # Update theta only if bounced
    if bounced:
        theta = math.atan2(pvy, pvx)
    
    return (px, py, theta)
    
@cuda.jit
def particle_pos_kernel(particles, screen, offsets, rng_states, out):
    i = cuda.grid(1)

    if i >= particles.shape[0]:
        return
    
    p = particles[i]
    r, g, b = p[3], p[4], p[5]

    random_theta = random.xoroshiro128p_uniform_float32(rng_states, i)

    p[2] = update_theta(p, screen, offsets, random_theta)
    px, py, theta = update_pos(p)

    out[i] = px, py, theta, r, g, b

def compute_particle_pos(particles, screen_pixels):
    particles = np.array(particles)
    out = np.zeros_like(particles)

    arr_device = cuda.to_device(particles)
    out_device = cuda.device_array(particles.shape)
    screen_pixels_device = cuda.to_device(screen_pixels)
    offsets_device = cuda.to_device(SAMP_OFFSETS)

    threads_per_block = 32
    blocks = int(np.ceil(particles.shape[0] / threads_per_block))

    # RNG states to turn each particle by a small random amount each frame
    rng_states = random.create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    
    particle_pos_kernel[blocks, threads_per_block](arr_device, screen_pixels_device, offsets_device, rng_states, out_device)
    out = out_device.copy_to_host()
    
    return out