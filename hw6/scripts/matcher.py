import numpy as np
from PIL import Image
import random

from harris import harris, save_marked
from utils import gradient_x, gradient_y, apply_gaussian

# returns dictionary in format: (x,y) location of corner -> (Gx, Gy) gradients
def harris_keypoints(img, corners):
	img = np.asarray(Image.open(img).convert('L'))
	keypoints = {}
	for x,y in corners:
		window = img[x-2:x+3, y-2:y+3]
		Gx = gradient_x(window)
		Gy = gradient_y(window)
		keypoints[(x,y)] = (Gx, Gy)
	return keypoints

# finds best correspondenses in both images given keypoints of images
# ONLY returns bottom half a.k.a closest L2 distance-wise
# return format is a dictionary: ((x1,y1), (x2,y2)=best_match_to_p1_in_kp2) -> distance
def match(kp1, kp2):
	matches = {}
	for p1 in kp1:
		best_distance = 1000000000000000
		best_point = None
		Gx1, Gy1 = kp1[p1]
		for p2 in kp2:
			Gx2, Gy2 = kp2[p2]
			if Gx1.shape != Gx2.shape or Gy1.shape != Gy2.shape: continue
			norm1 =np.linalg.norm(Gx1-Gx2)
			norm2 =np.linalg.norm(Gy1-Gy2) 
			distance = norm1+norm2
			if distance < best_distance: 
				best_distance, best_point = distance, p2

		matches[(p1, best_point)] = best_distance
	distances = list(matches.values())
	bottom = np.sort(distances)[:int(len(distances)/2)] # bottom half, returns DISTANCES
	matches = {key:val for key, val in matches.items() if val in bottom}
	return matches

def get_matches(im1, im2):
	im1_corners, im2_corners = harris(im1, save=True, saveloc="lpoints.png"), harris(im2, save=True, saveloc="rpoints.png")

	im1_kps = harris_keypoints(im1, im1_corners)
	im2_kps = harris_keypoints(im2, im2_corners)

	return match(im1_kps, im2_kps)