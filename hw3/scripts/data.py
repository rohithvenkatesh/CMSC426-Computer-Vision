import numpy as np
import glob
from PIL import Image, ImageDraw

import hog

# path: path to photos folder
def load_data(path):
	X = []
	set_type = path.split('/')[-3]
	class_type = path.split('/')[-2]
	count = 1
	total = len(glob.glob(path))
	for im in glob.glob(path):
		print(str(count) + '/' + str(total) + ' —— ' + str(im))
		hog_vector = hog.HOG(im)
		X.append(hog_vector)
		count+=1
	X = np.array(X)
	return X

def full_data():
	# DATA PATH #
	dp = '/Users/rohithvenkatesh/Downloads/hw3-dataset/'

	# TRAIN DATA #
	train_pos, train_neg = load_data(dp + 'train/pos/*.png'), load_data(dp + 'train/neg/*.png')
	np.save('../data/train_pos', train_pos)
	np.save('../data/train_neg', train_neg)

	# TEST DATA #
	test_pos, test_neg = load_data(dp + 'test/pos/*.png'), load_data(dp + 'test/neg/*.png')
	np.save('../data/test_pos', test_pos)
	np.save('../data/test_neg', test_neg)

full_data()