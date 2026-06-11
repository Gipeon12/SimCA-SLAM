import os
import argparse
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise


### ARGUMENTS PARSING ###

parser = argparse.ArgumentParser(allow_abbrev = False)
parser.add_argument("--imgsize", type = int,   default = 900,  help = "Image size in pixels")
parser.add_argument("--bordmin", type = int,   default = 5,    help = "Minimum border extent from the image edge")
parser.add_argument("--bordmax", type = int,   default = 25,   help = "Maximum border extent from the image edge")
parser.add_argument("--rmunder", type = int,   default = 30,   help = "Size below which an object is removed")
parser.add_argument("--densval", type = float, default = 0.35, help = "Overshoot ratio to turn a pixel into full occupancy")
parser.add_argument("--anisogn", action='store_true',          help = "Trigger for a ground generation with spatial disparity")
parser.add_argument("--noblank", action='store_false',         help = "Trigger for not clearing the central area of the arena")
args = parser.parse_args()


###  RANDOM NOISE GENERATION  ###

def generate_perlin_noise(size = 900):
    # Generate a large perlin noise as a square grid of a given size.
    # Add a first noise to the transposition of a second noise to absorb any spatial repetition.
    s1 = rd.randint(1, 1000)
    s2 = rd.randint(1001, 2000) # We ensure there is no chance for the seeds to be the same.
    noise1 = PerlinNoise(octaves = 20, seed = s1)
    noise2 = PerlinNoise(octaves = 20, seed = s2)
    subpic = [[noise1([i/size, j/size]) for j in range(size)] for i in range(size)]
    suppic = [[noise2([i/size, j/size]) for j in range(size)] for i in range(size)]
    raw_noise = [[subpic[i][j]+suppic[j][i] for j in range(size)] for i in range(size)]
    seed_tag = f"{s1}t{s2}"
    print(f"Perlin noise of size {size} generated with seed {seed_tag}.")
    return raw_noise, seed_tag


###  GRID OPERATIONS  ###

def normalize(raw_noise):
    # Scale all noise values between 0 and 1.
    maxima = [max(row) for row in raw_noise]
    minima = [min(row) for row in raw_noise]
    M, m = max(maxima), min(minima)
    normal_noise = []
    for row in raw_noise:
        normal_noise += [[(val-m)/(M-m) for val in row],]
    return normal_noise

def sharpen_contrast(normal_noise):
    # Sharpen the contrast between empty and occupied areas using the hyperbolic tangent.
    sharp_noise = []
    for row in normal_noise:
        sharp_noise += [[(1 + np.tanh(10*val-5))/2 for val in row],]
    return sharp_noise

def binarize(normal_noise, density_value, filter_matrix):
    # Convert noise into binary values using an overshoot ratio to control obstacle density.
    N, P = len(normal_noise), len(normal_noise[0])
    binar_noise = []
    for i in range(N):
        row = []
        for j in range(P):
            val, filter_value = normal_noise[i][j], filter_matrix[i][j]
            row += [int( (val + density_value) * filter_value ),]
        binar_noise += [row,]        
    return binar_noise


###  ARENA PROCESSING  ###

def remove_undersized_blocks(binar_noise, min_size = 30):
    # Remove all obstacles smaller than a certain size.
    N, P = len(binar_noise), len(binar_noise[0])
    binar_noise = np.array(binar_noise)
    free_stamp = np.zeros((min_size, min_size))
    edges_mask = np.zeros_like(free_stamp, dtype = bool)
    edges_mask[1:-1,1:-1] = True
    for i in range(N-min_size):
        for j in range(P-min_size):
            cell_ij = binar_noise[i:i+min_size, j:j+min_size]
            null_edges = (cell_ij == free_stamp) | edges_mask
            if null_edges.all():
                binar_noise[i:i+min_size, j:j+min_size] = free_stamp
    print(f"Removed blocks under size: {min_size} pixels.")
    return binar_noise

def build_marginal_walls(binar_noise, sharp_noise, bord_limits = (5, 25)):
    # Create impenetrable borders with irregularities derived from marginal noise.
    N, P = len(binar_noise), len(binar_noise[0])
    (exmin, exmax) = bord_limits
    extent = exmax - exmin
    for i in range(N):
        jL = exmin + int(sharp_noise[i][exmax] * extent)
        jR = (P - 1 - exmin) - int(sharp_noise[i][-exmax] * extent)
        bord_cols = [j for j in range(jL)] + [j for j in range(jR, P)]
        for j in bord_cols:
            binar_noise[i][j] = 1
    for j in range(P):
        iT = exmin + int(sharp_noise[exmax][j] * extent)
        iB = (N - 1 - exmin) - int(sharp_noise[-exmax][j] * extent)
        bord_rows = [i for i in range(iT)] + [i for i in range(iB, N)]
        for i in bord_rows:
            binar_noise[i][j] = 1
    print("Added plain borders from noise.")
    return binar_noise


###  IMAGE SAVING  ###

def save_image(arena, seed_tag, base_dir):
    invpic = np.ones_like(arena) - arena
    save_dir = os.path.join(base_dir, "arenas")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{seed_tag}.png")
    plt.imsave(save_path, invpic, cmap="gray")
    print(f"Image created and saved at {save_path}.")
    

###  GENERATOR  ###

class Arena2D:

    def __init__(self, params = args):
        self.img_size  = params.imgsize
        self.dens_val  = params.densval
        self.anisoGnd  = params.anisogn
        self.blankCen  = params.noblank
        self.bord_min  = params.bordmin
        self.bord_max  = params.bordmax
        self.rm_under  = params.rmunder
        self.cwd       = os.getcwd()
    
    def generate(self):
        (raw_noise, seed_tag) = generate_perlin_noise(self.img_size)
        normal_noise = normalize(raw_noise)
        sharp_noise = sharpen_contrast(normal_noise)
        filter_matrix = np.ones([self.img_size, self.img_size])
        disp_msg = ""
        if self.blankCen:
            mask = []
            hole_rad = self.img_size/10  # Central area radius in pixels (left empty)
            for i in range(self.img_size):
                row = []
                for j in range(self.img_size):
                    val = 1
                    if ((self.img_size/2-i)**2+(self.img_size/2-j)**2 <= hole_rad**2):
                        val = ((self.img_size/2-i)**2+(self.img_size/2-j)**2)**0.5/hole_rad
                    row += [val,]
                mask += [row,]
            filter_matrix = sharpen_contrast(mask)
        if self.anisoGnd:
            s = rd.randint(2001,3000)
            noise = PerlinNoise(octaves = 2, seed = s)
            f_noise = [[noise([i/self.img_size, j/self.img_size]) for j in range(self.img_size)] for i in range(self.img_size)]
            seed_tag = f"{seed_tag}f{s}"
            print(f"Density filter noise generated with seed {s}.")
            nf_noise = normalize(f_noise)
            filter_matrix = np.multiply(np.array(sharpen_contrast(nf_noise)), filter_matrix)
            disp_msg = " spatial disparity and"
        arena = binarize(normal_noise, self.dens_val, filter_matrix)
        arena = remove_undersized_blocks(arena, self.rm_under)
        arena = build_marginal_walls(arena, sharp_noise, bord_limits = (self.bord_min, self.bord_max))
        print(f"Arena generated from perlin noise with{disp_msg} density set on: {self.dens_val}.")        
        save_image(arena, seed_tag, self.cwd)


if __name__ == '__main__':
    a2d = Arena2D()
    a2d.generate()
