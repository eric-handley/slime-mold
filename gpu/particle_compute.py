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

@cuda.jit
def sum_sample_pixels(sx, sy, screen, offsets, rgb, settings):
    sum = 0
    r, g, b = rgb
    c = max(r, g, b)

    for i in range(offsets.shape[0]):
        x, y = offsets[i]
        spx = sx + x
        spy = sy + y

        if spx > 0 and spy > 0 and spx < settings[0] and spy < settings[1]:
            sr, sg, sb = screen[int(spx), int(spy)]
            sc = max(sr, sg, sb)
            
            # If max channel of sampled pixel matches max channel of particle
            if (sc == sr and c == r) or (sc == sb and c == b) or (sc == sg and c == g):
                sum += settings[13]
            if (sc == sr and c != r) or (sc == sb and c != b) or (sc == sg and c != g):
                sum += settings[12]
    
    return sum

@cuda.jit
def update_theta(p, screen, offsets, random_theta, settings):
    px, py, theta, r, g, b = p

    # Sample three points in front of particle
    sx1 = px + math.cos(theta + settings[3]) * settings[5]
    sy1 = py + math.sin(theta + settings[3]) * settings[5]

    sx2 = px + math.cos(theta) * settings[5]
    sy2 = py + math.sin(theta) * settings[5]

    sx3 = px + math.cos(theta + settings[4]) * settings[5]
    sy3 = py + math.sin(theta + settings[4]) * settings[5]

    s1 = sum_sample_pixels(sx1, sy1, screen, offsets, (r,g,b), settings)
    s2 = sum_sample_pixels(sx2, sy2, screen, offsets, (r,g,b), settings)
    s3 = sum_sample_pixels(sx3, sy3, screen, offsets, (r,g,b), settings)

    # # Find highest weighted point
    smax = max(s1, s2, s3)

    a, b = -settings[11], settings[11]
    theta += a + (b - a) * random_theta

    # Check for tie, otherwise we will bias turning left when samples are equal
    if (smax == s1 and smax == s2) or (smax == s2 and smax == s3) or (smax == s1 and smax == s2):
        return theta

    theta_left = theta + settings[3]
    theta_right = theta + settings[4]   

    # Turn toward highest sampled area
    if s1 == smax:
        return theta_left
    if s3 == smax:
        return theta_right
        
    return theta


@cuda.jit
def update_pos(p, settings):
    px, py, theta = p[0], p[1], p[2]

    # Find velocity based on current theta, update position by that velocity
    pvx = math.cos(theta) * settings[7]
    pvy = math.sin(theta) * settings[7]
    
    nx = px + pvx
    ny = py + pvy
    
    bounced = False
    
    # Particle "bounces" if it reaches edge of screen
    if nx < 1:
        px = 2
        pvx = abs(pvx)
        bounced = True
    elif nx >= settings[0]:
        px = settings[0] - 2
        pvx = -abs(pvx) + 0.5
        bounced = True
    else:
        px = nx
    
    if ny < 1:
        py = 2
        pvy = abs(pvy)
        bounced = True
    elif ny >= settings[1]:
        py = settings[1] - 2
        pvy = -abs(pvy) + 0.5
        bounced = True
    else:
        py = ny
    
    # Update theta only if bounced
    if bounced:
        theta = math.atan2(pvy, pvx)
    
    return (px, py, theta)
    
@cuda.jit
def particle_pos_kernel(particles, screen, offsets, rng_states, settings, out):
    i = cuda.grid(1)

    if i >= particles.shape[0]:
        return
    
    p = particles[i]
    r, g, b = p[3], p[4], p[5]

    random_theta = random.xoroshiro128p_uniform_float32(rng_states, i)

    p[2] = update_theta(p, screen, offsets, random_theta, settings)
    px, py, theta = update_pos(p, settings)

    out[i] = px, py, theta, r, g, b

def compute_particle_pos(particles, screen_pixels, settings: Settings):
    particles = np.array(particles)
    out = np.zeros_like(particles)

    arr_device = cuda.to_device(particles)
    out_device = cuda.device_array(particles.shape)
    screen_pixels_device = cuda.to_device(screen_pixels)
    offsets_device = cuda.to_device(generate_pixel_offsets(settings.SAMPLE_RADIUS))
    settings_device = cuda.to_device(get_settings_arr(settings))

    threads_per_block = 32
    blocks = int(np.ceil(particles.shape[0] / threads_per_block))

    # RNG states to turn each particle by a small random amount each frame
    rng_states = random.create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    
    particle_pos_kernel[blocks, threads_per_block](arr_device, screen_pixels_device, offsets_device, rng_states, settings_device, out_device)
    out = out_device.copy_to_host()
    
    return out

def get_settings_arr(settings: Settings):
    # Yes this is awful, I'm sorry
    return [
        settings.SURFACEX,
        settings.SURFACEY,
        settings.SAMPLE_ANGLE,
        -1 * settings.SAMPLE_ANGLE,
        settings.SAMPLE_ANGLE,
        settings.SAMPLE_DISTANCE,
        settings.SAMPLE_RADIUS,
        settings.VELOCITY,
        settings.COHESION,
        settings.TURN_WEIGHT_LEFT,
        settings.TURN_WEIGHT_RIGHT,
        settings.TURN_RANDOMNESS,
        settings.AVOID_WEIGHT,
        settings.ATTRACT_WEIGHT
    ]