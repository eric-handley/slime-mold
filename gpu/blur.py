import numpy as np
from numba import cuda
from settings import Settings

def generate_blur_offsets(size):
    offsets = []
    for x in range(-size, size + 1):
        for y in range(-size, size + 1):
            offsets.append((x, y))
    return offsets

@cuda.jit
def blur_kernel(arr, offsets, out, decay_speed):
    # Simple box blur
    x, y = cuda.grid(2)

    if x >= arr.shape[0] or y >= arr.shape[1]:
        return

    r_sum, g_sum, b_sum, t = 0, 0, 0, 0

    # Sum each channel in 3x3 radius
    for n in range(offsets.shape[0]):
        i, j = offsets[n]
        if 0 <= x + i < arr.shape[0] and 0 <= y + j < arr.shape[1]:
            pixel = arr[x + i, y + j]
            r_sum += pixel[0]
            g_sum += pixel[1]
            b_sum += pixel[2]
            t += 1

    # Calculate the average for each channel
    r_avg = r_sum // t
    g_avg = g_sum // t
    b_avg = b_sum // t
    
    out[x, y, 0] = r_avg - decay_speed if r_avg > decay_speed else 0
    out[x, y, 1] = g_avg - decay_speed if g_avg > decay_speed else 0
    out[x, y, 2] = b_avg - decay_speed if b_avg > decay_speed else 0

def blur(arr, settings: Settings):
    arr = np.array(arr, dtype=np.uint8)
    out = np.zeros_like(arr)
    decay_speed = settings.DECAY_SPEED

    arr_device = cuda.to_device(arr)
    offsets_device = cuda.to_device(generate_blur_offsets(1))
    out_device = cuda.device_array(arr.shape)

    threads_per_block = (32, 32)
    blocks_per_grid_x = int(np.ceil(arr.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(arr.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    blur_kernel[blocks_per_grid, threads_per_block](arr_device, offsets_device, out_device, decay_speed)
    out = out_device.copy_to_host()
    
    return out