import math as m
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal as sig
import random

from transformation import estimate_transformation, compute_error
from matcher import get_matches
from ransac import RANSAC, flipxy

def main(im1loc, im2loc, output):
	correspondences = flipxy(get_matches(im1loc,im2loc))
	model = RANSAC(correspondences)

	im1, im2 = Image.open(im1loc).convert('L'), Image.open(im2loc).convert('L')
	w1,h1 = im1.size
	w2, h2 = im2.size
	panorama = im2.transform((w1+w2,h1), Image.PERSPECTIVE, model, Image.BILINEAR)
	panorama.save("transformed.jpg")

	im1_arr, panorama_arr = np.asarray(im1), np.asarray(panorama)
	final = np.copy(panorama_arr)
	for i in range(len(panorama_arr)):
		for j in range(len(panorama_arr[i])):
			if j < len(im1_arr[i]):
				pixel1 = im1_arr[i,j]
				pixel2 = panorama_arr[i,j]
				if pixel2 == 0 and pixel1 != 0: final[i,j] = pixel1
				if pixel2 == 0 and pixel1 == 0: final[i,j] = 0
				if pixel2 != 0 and pixel1 != 0: final[i,j] = pixel2 #(int(pixel1) + int(pixel2))/2
				if pixel2 != 0 and pixel1 == 0: final[i,j] = pixel2
	test = Image.fromarray(np.uint8(final)).convert('L')
	test.save(output)

left = input('enter LEFT input image (cannot swap left and right images :/): \n')
right = input('enter RIGHT input image: \n')
output = input('enter output name to save to: \n')
main(left, right, output)