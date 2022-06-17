import taichi as ti
import os
import math

def run_poisson(desired_samples = 100000):
    ti.init(arch=ti.cpu)

    run_test = False
    output_frames = False

    grid_n = 400
    res = (grid_n, grid_n)
    dx = 1 / res[0]
    inv_dx = res[0]
    radius = dx * math.sqrt(2)
    
    grid = ti.field(dtype=int, shape=res)
    samples = ti.Vector.field(2, dtype=float, shape = desired_samples)

    @ti.kernel
    def random_sample(deisred_samples: int):
        if True:
            for i in range(desired_samples):
                samples[i] = ti.Vector([ti.random(), ti.random()])
    
    @ti.func
    def place_sample(sample_id, x):
        grid_index = int(ti.floor(x * inv_dx))
        grid[grid_index] = sample_id
    

    @ti.kernel
    def poisson_disk_sample(desired_samples: int) -> int:
        samples[0] = ti.Vector([0.5, 0.5])
        