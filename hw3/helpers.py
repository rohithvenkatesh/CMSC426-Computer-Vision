import numpy as np

def convolve(im_arr, f, pad=False):
	img_out = np.zeros(im_arr.shape)

	if len(f.shape) != 1: f.shape = (f.shape[0], f.shape[1])
	else: f.shape = (f.shape[0], 1)

	fltr_sz_x = f.shape[0]
	fltr_sz_y = f.shape[1]
	if (pad):
		for i in range(int(fltr_sz_y/2)):
			im_arr = np.append(im_arr, im_arr[:,[-1]], axis=1)
			im_arr = np.insert(im_arr, 0, im_arr[:,0], axis=1)
		for i in range(int(fltr_sz_x/2)):
			im_arr = np.append(im_arr, im_arr[[-1],:], axis=0)
			im_arr = np.insert(im_arr, 0, im_arr[0,:], axis=0)

	for i in range(1, im_arr.shape[0]-fltr_sz_y-1):
		for j in range(1, im_arr.shape[1]-fltr_sz_x-1):
			value =  np.multiply(f, im_arr[(i - 1):(i + fltr_sz_x-1), (j - 1):(j + fltr_sz_y-1)])
			img_out[i, j] = value.sum ()

	return img_out.astype(np.float32)

def gradient_calc(img):
	Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
	Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

	Kx = np.array([[-0.5, 0, 0.5], [-0.5, 0, 0.5], [-0.5, 0, 0.5]], np.float32)
	Ky = np.array([[-0.5, -0.5, -0.5], [0, 0, 0], [0.5, 0.5, 0.5]], np.float32)

	J_x = convolve(img, Kx)
	J_y = convolve(img, Ky)

	E_s = np.sqrt(np.add(J_x**2, J_y**2))
	E_o = np.arctan2(J_y, J_x) * 180 / np.pi
	E_o[E_o < 0] += 180
	return E_s, E_o