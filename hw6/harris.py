import numpy as np
from scipy import signal as sig
from PIL import Image, ImageDraw
import math as m
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

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
	# return convolve(im, kernel_x, pad=True)
	return sig.convolve2d(im, kernel_x, mode='same')

def gradient_y(im):
	kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	# return convolve(im, kernel_y, pad=True)
	return sig.convolve2d(im, kernel_y, mode='same')

def apply_gaussian(k, sigma, img):
	fltr_sz = 2*k + 1
	gaussian = np.zeros((fltr_sz, fltr_sz))
	for i in range(1, fltr_sz+1):
		for j in range(1, fltr_sz+1):
			f = 1.0 / (2 * np.pi * sigma**2)
			q = np.exp(- ( (i-(k+1))**2 + (j-(k+1))**2 ) / (2 * sigma**2))
			gaussian[i-1,j-1] = f*q
	# return convolve(img, gaussian, pad=False)
	return sig.convolve2d(img, gaussian, mode='same')

def mark_corners(img, R):
	corners = np.copy(img)

	detections = {}
	for rowindex, response in enumerate(R):
		for colindex, r in enumerate(response):
			detections[r] = (rowindex,colindex)
	keys = list(detections.keys())
	top = np.sort(keys)[-int(len(keys)/100):]
	for i in top:
		row, col = detections[i][0], detections[i][1]
		corners[row,col] = [255,0,0]
	return corners

imcolor = Image.open("e.png").convert('RGB')
im = np.asarray(imcolor.convert('L'))

## using scipy convolve
I_x = gradient_x(im)
I_y = gradient_y(im)
Ixx = apply_gaussian(3, 1, I_x**2)
Ixy = apply_gaussian(3, 1, I_y*I_x)
Iyy = apply_gaussian(3, 1, I_y**2)

# Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
# Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
# Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

k = 0.06
detM = Ixx * Iyy - Ixy ** 2
traceM = Ixx + Iyy
R = detM - k * traceM 

imcolor = mark_corners(imcolor, R)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
ax.set_title("corners found")
ax.imshow(imcolor, cmap='gray')
plt.show()
