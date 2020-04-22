from PIL import Image, ImageDraw
import argparse
import test2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from test2 import CNN, dataset 
from torch.autograd import Variable

def get_crops(im, scale_factor):
	crops = []
	for y in range(0,im.size[0],16):
		for x in range(0, im.size[1], 16):
			y0, x0, y1, x1 = float(y/scale_factor), float(x/scale_factor),float((y+64)/scale_factor), float((x+128)/scale_factor)
			im1 = im.crop((y, x, y+64, x+128))
			crops.append((im1, (y0, x0, y1, x1)))
	return crops

def multiscale(image):
	imw, imh = image.size
	crops = []
	scale_factor = 1
	factor = 0.80
	while imw >= 64 and imh >= 128:
		crops += get_crops(image, scale_factor) 
		imw, imh = image.size
		image = image.resize(( int(round(imw*factor)), int(round(imh*factor)) ))
		scale_factor = scale_factor * factor
	return crops

def detect_pedestrians(image, output_name, ped_threshold, nms_threshold):
	crops = multiscale(image)
	locations, boxes, scores = [], [], []
	for img, location in crops:
		to_tensor = transforms.ToTensor()
		img_tensor = Variable(to_tensor(img).float(), requires_grad=True).unsqueeze(0)
		pedestrian_prob = F.softmax(model(img_tensor), dim=1).data.tolist()[0][1]
		if pedestrian_prob > ped_threshold:
			locations.append(location)
			boxes.append(location)
			scores.append(pedestrian_prob)
			# ImageDraw.Draw(image, 'RGBA').rectangle(location, outline='green', fill=(0, 150, 0, 60))
	boxes = torch.tensor(boxes)
	scores = torch.tensor(scores)
	selected = torchvision.ops.nms(boxes, scores, nms_threshold).tolist()
	for i in selected:
		ImageDraw.Draw(image, 'RGBA').rectangle(locations[i], outline='green', fill=(0, 150, 0, 60))
	image.save(output_name)

## RUN WITH WEIGHTS, INPUT, OUTPUT FLAGS (--weights, --input, --output)
## EXAMPLE: python evaluate.py --weights 'trained_model.pt' --input 'pic.jpg' --output 'save.jpg'     
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='code')
	parser.add_argument('--weights', type=str, default='trained_model.pt', help='CNN weights')
	parser.add_argument('--input', type=str, default=None, help='Image to Scan')
	parser.add_argument('--output', type=str, default='detected_pedestrians.png', help='Image to Save to')
	args = parser.parse_args()

	model = CNN().eval()
	model.load_state_dict(torch.load(args.weights))
	image = Image.open(args.input).convert('RGB')
	detect_pedestrians(image, args.output, 0.99, 0.04) # can change pedestrian detection threshold and nms threshold for different results
