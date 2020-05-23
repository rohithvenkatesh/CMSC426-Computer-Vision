import math as m
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal as sig

from utils import gradient_x, gradient_y, apply_gaussian

# input matrix R of harris values
# returns a tuple array of (x,y) locations of corners
# divides total corner detections by 500 and returns top 0.002%
def get_corner_locations(R):
	detections = {}
	for rowindex, response in enumerate(R):
		for colindex, r in enumerate(response):
			detections[r] = (rowindex,colindex)

	# get top % of harris corners	
	keys = list(detections.keys())
	top = np.sort(keys)[-int(len(keys)/500):] # top %

	# delete all corners not in top %
	delete = [key for key in detections.keys() if key not in top]
	for key in delete: del detections[key]
	
	print("number of corners after filter: " + str(len(detections.keys())))
	return detections.values()

# convert to white background with red corners untouched
def cvt2wb(img):
	for i in range(len(img)):
			for j in range(len(img[i])):
				if (img[i,j] != [255,0,0]).all(): img[i,j] = [255,255,255]

# saves the image with the filtered corners marked into a file name in output variable
def save_marked(img, corner_locations, saveloc):
	img = np.copy(img)
	for x,y in corner_locations: img[x,y] = [255,0,0]
	# cvt2wb(img)
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
	ax.set_title("corners found")
	ax.imshow(img, cmap='gray')
	plt.savefig(saveloc)

## returns image with marked corners
def harris(img, k=0.06, save=False, saveloc="marked_image.png"):
	imcolor = Image.open(img).convert('RGB')
	im = np.asarray(imcolor.convert('L'))

	I_x, I_y = gradient_x(im), gradient_y(im)
	Ixx, Ixy, Iyy = apply_gaussian(3, 1, I_x**2), apply_gaussian(3, 1, I_y*I_x), apply_gaussian(3, 1, I_y**2)

	detM = Ixx * Iyy - Ixy ** 2
	traceM = Ixx + Iyy
	R = detM - k * traceM 

	harris_corners = get_corner_locations(R)
	if save: save_marked(imcolor, harris_corners, saveloc) # save image with marked corners
	return harris_corners