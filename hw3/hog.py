import numpy as np
from PIL import Image, ImageDraw
import math as m
import matplotlib.pyplot as plt

import helpers as hp

## GLOBALS ##
IMG_SIZE = 400
N_BUCKETS = 9
CELL_SIZE = 8

## GET HISTOGRAM OF ONE CELL ##
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

## GENERATE CELL MATRIX ##
def make_cells(M, D):
	n = int(IMG_SIZE / CELL_SIZE)
	cells = np.empty((n, n), dtype=np.ndarray)
	for i in range(0, M.shape[0], CELL_SIZE):
		for j in range(0, M.shape[1], CELL_SIZE):
			cell_M = M[i:i+CELL_SIZE, j:j+CELL_SIZE]
			cell_D = D[i:i+CELL_SIZE, j:j+CELL_SIZE]
			cells[int(i/CELL_SIZE), int(j/CELL_SIZE)] = get_cell_histogram(cell_M, cell_D)
	return cells

## GET HOG FEATUER VECTOR ##
def hog_feature_vector(cells):
	vectors = []
	for i in range(0, cells.shape[0], 2):
		for j in range(0, cells.shape[1], 2):
			vector = np.concatenate((cells[i,j], cells[i,j+1], cells[i+1,j], cells[i+1, j+1]))
			vector = vector / np.linalg.norm(vector)
			vectors.append(vector)
	return np.array(vectors)

## MAIN ##
def HOG(img_name):
	im = Image.open(img_name).resize((IMG_SIZE,IMG_SIZE)).convert('L')
	img = np.array(im).astype(np.float32)

	M, D = hp.gradient(img)
	cells = make_cells(M,D)
	histogram = hog_feature_vector(cells)
	return histogram