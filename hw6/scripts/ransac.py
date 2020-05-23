import math as m
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal as sig
import random

from transformation import estimate_transformation, compute_error
from matcher import get_matches

# takes as input an array of 2-tuples in format: [((x1,y1), (x2,y2))...]
def RANSAC(correspondences, n=6, k=300, t=3, d=15):
	best_model, best_err = None, 100000000000
	for i in range(k):
		maybe_inliers = random.sample(correspondences, n)
		pointsa, pointsb = zip(*maybe_inliers)
		maybe_model = estimate_transformation(pointsa, pointsb)
		also_inliers = []
		for c in list(set(correspondences) - set(maybe_inliers)): # correspondence not in maybeInliers
			if compute_error([c[0]], maybe_model, [c[1]]) < t:
				also_inliers.append(c)
		
		if len(also_inliers) > d:
			also_pointsa, also_pointsb = zip(*also_inliers)
			finala, finalb = pointsa + also_pointsa, pointsb + also_pointsb
			better_model = estimate_transformation(finala, finalb)
			err = compute_error(finala, better_model, finalb)
			if err < best_err:
				best_model, best_err = better_model, err

	return best_model

# Flips the (x,y) tuple to work with transformation.py
def flipxy(correspondences):
	flipped_correspondences = []
	for c in correspondences.keys():
		new_tup = (tuple(reversed(c[0])), tuple(reversed(c[1])))
		flipped_correspondences.append(new_tup)
	return flipped_correspondences

