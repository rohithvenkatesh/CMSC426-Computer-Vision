import numpy as np
from PIL import Image, ImageDraw
import math as m
import matplotlib.pyplot as plt

import helpers as hp
import hog

DEFAULT_IMG = "input2.jpg" 
im = input("Enter image name (or leave empty for default):\n") or DEFAULT_IMG

hog_vector = hog.HOG(im)


print(hog_vector.shape)