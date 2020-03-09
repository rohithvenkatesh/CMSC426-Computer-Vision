import numpy as np
import glob
from PIL import Image, ImageDraw

import helpers as hp
import hog

X_train_pos, y_train_pos = np.load('../data/X_train_pos.npy'), np.load('../data/y_train_pos.npy')
X_train_pos = np.nan_to_num(X_train_pos, nan=0.01)

print(X_train_pos.shape, y_train_pos.shape)

# datapoints is X_train | labels is y_train | w is weights vector of # of datapoints | gamma is learning rate 
def learn(datapoints, labels, gamma, epochs):
	N = len(datapoints)
	w = np.zeros(len(datapoints[0]))-1
	for _ in range(epochs): # epochs	
		for k in range(N):
			if labels[k] * np.dot(w, datapoints[k]) < 1:
				w += gamma*labels[k]*datapoints[k]
	return w

learned_weights = learn(X_train_pos, y_train_pos, 0.005, 100)
predictions = X_train_pos.dot(learned_weights).astype(np.float64)
print(len(predictions[predictions<=3]))