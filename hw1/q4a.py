import numpy as np
import PIL
from PIL import Image, ImageDraw

# part (a)
x = np.transpose(np.array([[20,20,1],[20,140,1],[60,60,1]]))
y = np.transpose(np.array([[230,130,1],[350,190,1],[290,110,1]]))
T = y.dot(np.linalg.inv(x))

# part (b)
im = Image.new('RGB', (255, 255))
draw = ImageDraw.Draw(im)
draw.polygon([(20,20),(20,140),(60,60)])
im.save('triangle1.png')

# part (c)
Tinv = np.linalg.inv(T)
data =[Tinv[0,0], Tinv[0,1], Tinv[0,2], Tinv[1,0], Tinv[1,1], Tinv[1,2]]
im1 = im.transform((500,500), PIL.Image.AFFINE, data)
im1.save('triangle2.png')

