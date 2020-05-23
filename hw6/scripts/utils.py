import math as m
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal as sig

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

def gradient_x(im):
	kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
	return convolve(im, kernel_x, pad=True)
	# return sig.convolve2d(im, kernel_x, mode='same', boundary='symm')

def gradient_y(im):
	kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	return convolve(im, kernel_y, pad=True)
	# return sig.convolve2d(im, kernel_y, mode='same', boundary='symm')

def apply_gaussian(k, sigma, img):
	fltr_sz = 2*k + 1
	gaussian = np.zeros((fltr_sz, fltr_sz))
	for i in range(1, fltr_sz+1):
		for j in range(1, fltr_sz+1):
			f = 1.0 / (2 * np.pi * sigma**2)
			q = np.exp(- ( (i-(k+1))**2 + (j-(k+1))**2 ) / (2 * sigma**2))
			gaussian[i-1,j-1] = f*q
	return convolve(img, gaussian, pad=False)
	# return sig.convolve2d(img, gaussian, mode='same')
