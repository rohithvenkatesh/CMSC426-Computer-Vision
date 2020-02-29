import numpy as np
import PIL
from PIL import Image, ImageDraw
import math as m
import matplotlib.pyplot as plt

import helpers

im = Image.open("./input.jpg")
# print(im.size)
im = im.convert('L')
img = np.array(im).astype(np.float32)

M, D = helpers.gradient_calc(img)

img_out = Image.fromarray(np.uint8(M)).convert('L')
img_out.save("output.jpg")
