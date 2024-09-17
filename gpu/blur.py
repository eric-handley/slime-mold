import numpy as np
from numba import cuda

@cuda.jit
def blur_kernel(arr, out):
    # Simple box blur
    x, y = cuda.grid(2)

    if x >= arr.shape[0] or y >= arr.shape[1]:
        return
    
    r_sum, g_sum, b_sum, t = 0, 0, 0, 0

    # Sum each channel in 3x3 radius
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= x + i < arr.shape[0] and 0 <= y + j < arr.shape[1]:
                pixel = arr[x + i, y + j]
                r_sum += pixel[0]
                g_sum += pixel[1]
                b_sum += pixel[2]
                t += 1

    # Calculate the average for each channel
    out[x, y, 0] = r_sum // t
    out[x, y, 1] = g_sum // t
    out[x, y, 2] = b_sum // t

def blur(arr):
    arr = np.array(arr, dtype=np.uint8)
    out = np.zeros_like(arr)

    arr_device = cuda.to_device(arr)
    out_device = cuda.device_array(arr.shape)

    threads_per_block = (32, 32)
    blocks_per_grid_x = int(np.ceil(arr.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(arr.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    blur_kernel[blocks_per_grid, threads_per_block](arr_device, out_device)
    out = out_device.copy_to_host()
    
    return out