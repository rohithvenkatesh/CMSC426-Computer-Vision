import numpy as np
import PIL
from PIL import Image, ImageDraw
import math as m
import matplotlib.pyplot as plt

import helpers as hp

IMG_SIZE = 400
im = Image.open("input2.jpg").resize((IMG_SIZE,IMG_SIZE)).convert('L')
img = np.array(im).astype(np.float32)

M, D = hp.gradient_calc(img)
# M, D = M.astype(np.uint16), D.astype(np.uint16)

N_BUCKETS = 9
CELL_SIZE = 8
def get_cell_histogram(cell_M, cell_D):
	buckets = np.linspace(0, 180, N_BUCKETS + 1)
	bucket_vals = np.zeros(N_BUCKETS)
	for i in range(0, CELL_SIZE):
		for j in range(0, CELL_SIZE):
			m = cell_M[i,j]
			d = cell_D[i,j]
			left_bin = int(d / 20.)
			right_bin = (int(d / 20.) + 1) % N_BUCKETS
			right_val = m * (d - left_bin * 20) / 20
			left_val= m - right_val
			bucket_vals[left_bin] += left_val
			bucket_vals[right_bin] += right_val
	return bucket_vals

def make_cells(M, D):
	n = int(IMG_SIZE / CELL_SIZE)
	cells = np.empty((n, n), dtype=np.ndarray)
	for i in range(0, M.shape[0], CELL_SIZE):
		for j in range(0, M.shape[1], CELL_SIZE):
			cell_M = M[i:i+CELL_SIZE, j:j+CELL_SIZE]
			cell_D = D[i:i+CELL_SIZE, j:j+CELL_SIZE]
			cells[int(i/CELL_SIZE), int(j/CELL_SIZE)] = get_cell_histogram(cell_M, cell_D)
	return cells

cells = make_cells(M,D)
img_out = Image.fromarray(np.uint8(M)).convert('L').save('output.jpg')