import numpy as np
import glob
from PIL import Image, ImageDraw
import pickle

import helpers as hp
import hog

# DEFAULT_IMG = "input2.jpg" 
# im = input("Enter image name (or leave empty for default):\n") or DEFAULT_IMG

# path: path to photos folder
# classification: True for pos and False for neg
def load_data(path):
	X = []
	set_type = path.split('/')[-3]
	class_type = path.split('/')[-2]
	for im in glob.glob(path):
		print(im)
		hog_vector = hog.HOG(im)
		print(hog_vector.shape)
		X.append(hog_vector)
	X = np.array(X)
	y = np.ones(X.shape[0]) if class_type == 'pos' else -1*np.ones(X.shape[0])
	return X,y

## dataset ##
dataset_path = '/Users/rohithvenkatesh/Downloads/hw3-dataset/'
train_pos_path = dataset_path + 'train/pos/*.png' 
train_neg_path = dataset_path + 'train/neg/*.png' 
test_pos_path = dataset_path + 'test/pos/*.png' 
test_neg_path = dataset_path + 'test/neg/*.png' 

X_train_pos, y_train_pos = load_data(train_pos_path)
print(X_train_pos[0].shape, '\n', y_train_pos[0])