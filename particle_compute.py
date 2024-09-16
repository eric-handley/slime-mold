import math
import numpy as np
from numba import cuda

from screen import Screen

windowx = Screen.windowx
windowy = Screen.windowy

@cuda.jit
def update_vel(p, particles):
    px, py, pvx, pvy = p[0], p[1], p[2], p[3]

    return (px, py, pvx, pvy)

@cuda.jit
def update_pos(p):
    px, py, pvx, pvy = p[0], p[1], p[2], p[3]

    nx = px + pvx
    ny = py + pvy

    if nx > 0 and nx < windowx:
        px = round(nx)
    else:
        pvx *= -1
        px = round(px + pvx)
    
    if ny > 0 and ny < windowy:
        py = round(ny)
    else:
        pvy *= -1
        py = round(py + pvy)

    return (px, py, pvx, pvy)
    
@cuda.jit
def particle_pos_kernel(particles, out):
    i = cuda.grid(1)

    # Ensure thread indices are within bounds
    if i >= particles.shape[0]:
        return
    
    p = particles[i]
    p = update_pos(p)
    px, py, pvx, pvy = update_vel(p, particles)
    
    out[i, 0] = px
    out[i, 1] = py
    out[i, 2] = pvx
    out[i, 3] = pvy

def compute_particle_pos(particles):
    particles = np.array(particles)
    out = np.zeros_like(particles)

    # Transfer data to the GPU
    arr_device = cuda.to_device(particles)
    out_device = cuda.device_array(particles.shape)

    # Define thread block size and grid size
    threads_per_block = 256
    blocks_per_grid = int(np.ceil(particles.shape[0] / threads_per_block))

    particle_pos_kernel[blocks_per_grid, threads_per_block](arr_device, out_device)
    out = out_device.copy_to_host()
    
    return out