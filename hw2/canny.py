import numpy as np
import PIL
from PIL import Image, ImageDraw
import math as m
import matplotlib.pyplot as plt

im = Image.open("input.jpg")
im = im.convert('L')
img = np.array(im).astype(np.float32)

## CONVOLUTATION IMPLEMENTATION ##
def convolve(im_arr, f, pad=False):
	img_out = np.zeros(im_arr.shape)

	if len(f.shape) != 1: f.shape = (f.shape[0], f.shape[1])
	else: f.shape = (f.shape[0], 1)

	fltr_sz_x = f.shape[0]
	fltr_sz_y = f.shape[1]
	if (pad):
		for i in range(int(fltr_sz_y/2)):
			im_arr = np.append(im_arr, im_arr[:,[-1]], axis=1)
			im_arr = np.insert(im_arr, 0, im_arr[:,0], axis=1)
		for i in range(int(fltr_sz_x/2)):
			im_arr = np.append(im_arr, im_arr[[-1],:], axis=0)
			im_arr = np.insert(im_arr, 0, im_arr[0,:], axis=0)

	for i in range(1, im_arr.shape[0]-fltr_sz_y-1):
		for j in range(1, im_arr.shape[1]-fltr_sz_x-1):
			value =  np.multiply(f, im_arr[(i - 1):(i + fltr_sz_x-1), (j - 1):(j + fltr_sz_y-1)])
			img_out[i, j] = value.sum ()

	return img_out.astype(np.float32)

## GAUSSIAN SMOOTHING ##
def apply_gaussian(k, sigma, img):
	fltr_sz = 2*k + 1
	gaussian = np.zeros((fltr_sz, fltr_sz))
	for i in range(1, fltr_sz+1):
		for j in range(1, fltr_sz+1):
			f = 1.0 / (2 * np.pi * sigma**2)
			q = np.exp(- ( (i-(k+1))**2 + (j-(k+1))**2 ) / (2 * sigma**2))
			gaussian[i-1,j-1] = f*q
	return convolve(img, gaussian, pad=True)

## GRADIENT CALCULATION ##
def gradient_calc(img):
	Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
	Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

	Kx = np.array([[-0.5, 0, 0.5], [-0.5, 0, 0.5], [-0.5, 0, 0.5]], np.float32)
	Ky = np.array([[-0.5, -0.5, -0.5], [0, 0, 0], [0.5, 0.5, 0.5]], np.float32)

	J_x = convolve(img, Kx)
	J_y = convolve(img, Ky)

	E_s = np.sqrt(np.add(J_x**2, J_y**2))
	E_o = np.arctan2(J_y, J_x) * 180 / np.pi
	E_o[E_o < 0] += 180
	return E_s, E_o

## NON MAXIMAL SUPPRESSION ##
def non_max_sup(E_s, E_o):
	I = np.zeros_like(E_s)
	for i in range(1,I.shape[0]-1):
		for j in range(1,I.shape[1]-1):
			q = 255
			r = 255
			if (0 <= E_o[i,j] < 22.5) or (157.5 <= E_o[i,j] <= 180):
				q = E_s[i, j+1]
				r = E_s[i, j-1]
			elif (22.5 <= E_o[i,j] < 67.5):
				q = E_s[i+1, j-1]
				r = E_s[i-1, j+1]
			elif (67.5 <= E_o[i,j] < 112.5):
				q = E_s[i+1, j]
				r = E_s[i-1, j]
			elif (112.5 <= E_o[i,j] < 157.5):
				q = E_s[i-1, j-1]
				r = E_s[i+1, j+1]

			if (E_s[i,j] >= q) and (E_s[i,j] >= r):
				I[i,j] = E_s[i,j]
			else:
				I[i,j] = 0
	return I

## THRESHOLDING ##
def threshold(img, t):
	threshold = img.max() * t
	I = np.zeros_like(img)
	for i in range(1,img.shape[0]-1):
		for j in range(1,img.shape[1]-1):
			I[i,j] = 150 if img[i,j] > threshold else 0
	return I

## MAIN ##
S = 0.7
K = 5
t = 0.15
img = apply_gaussian(K, S, img)
E_s, E_o = gradient_calc(img)
img = non_max_sup(E_s, E_o)
img = threshold(img, t) 

im2 = Image.fromarray(np.uint8(img)).convert('L')
im2.save("output1.jpg")



