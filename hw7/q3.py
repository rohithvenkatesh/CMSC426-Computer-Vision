import numpy as np
from PIL import Image

from utils import gradient_x, gradient_y
# Pixel class to store the intensity as well as previous best location
class Pixel():
    def __init__(self, intensity, prev_x=None):
        self.intensity = intensity
        self.prev_x = prev_x

# returns a list of pixels which make up the best path from top to bottom
def best_path(M):
	costs = []
	costs.append([Pixel(intensity, None) for intensity in M[0]])

	for y in range(1, len(M)):
		curr = M[y]
		row_costs = []

		for x, intensity in enumerate(curr):
			left, right = max(x-1, 0), min(x+1, len(curr)-1)
			xrange = range(left, right+1) # because right needs to be inclusive

			minx = min(xrange, key=lambda x: costs[y - 1][x].intensity)
			minparent = Pixel(intensity + costs[y-1][minx].intensity, minx)
			row_costs.append(minparent)
		costs.append(row_costs)

	endpoint = min(range(len(costs[-1])), key=lambda x: costs[-1][x].intensity)
	path = []
	x = endpoint
	for y in range(len(costs)-1, -1, -1):
		path.append((y, x))
		x = costs[y][x].prev_x
	return path[::-1]

# deletes path (as list of tuples which are the indices) from matrix M
def delete_path(M, path):
	final = []
	for i in range(len(M)):
		row = []
		for j in range(len(M[i])):
			if (i,j) != path[i]: row.append(M[i,j])
		final.append(row)
	return np.array(final)

img_name = input('enter input image: \n')
output_name = input('enter output name to save to: \n')
imcolor = Image.open(img_name)
imbw = imcolor.convert('L')

imcolor_arr = np.array(imcolor).astype(np.float32)
imbw_arr = np.array(imbw).astype(np.float32)

Ix = gradient_x(imbw_arr)
Iy = gradient_y(imbw_arr)
Gm = np.sqrt(np.add(Ix*Ix, Iy*Iy))

# currently runs 300 times, reducing the width by 300 pixels, change this number for different width reduction
for i in range(300):
	path = best_path(Gm)
	Gm = delete_path(Gm, path)
	imcolor_arr = delete_path(imcolor_arr, path)
	print("iteration #" + str(i) + ": " + str(imcolor_arr.shape))

output = Image.fromarray(np.uint8(imcolor_arr))
output.save(output_name)