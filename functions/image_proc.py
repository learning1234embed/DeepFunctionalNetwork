from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

# Calculate and return the total variation for one or more images
def total_variation_init(input_size):

	if input_size % 2 != 0:
		input_size -= 1
	
	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []	
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)
	
	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []	
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)
	
	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	in_channels = remained / in_width
	#print('in_chaneels', in_channels)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	#print('input_dim', parameter['input_dim'])

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def total_variation(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim) 
	#print('x_reshaped', x_reshaped)
	y = tf.image.total_variation(x_reshaped)
	#print('y', y)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	if y_shape[0] == 1:
		y_reshaped = tf.reshape(y, [-1, y_shape[0]]) 
	elif y_shape[0] == None:
		y_reshaped = tf.reshape(y, [-1, 1]) 
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def total_variation_py(x):
	return None

# Transpose image(s) by swapping the height and width dimension
def img_transpose_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1
	
	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []	
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)
	
	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []	
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)
	
	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	in_channels = remained / in_width
	#print('in_chaneels', in_channels)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def img_transpose(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim) 
	#print('x_reshaped', x_reshaped)
	y = tf.image.transpose(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])]) 
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def img_transpose_py(x):
	return None


# Flip an image horizontally (left to right)
def flip_h_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1
	
	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []	
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)
	
	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []	
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)
	
	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	in_channels = remained / in_width
	#print('in_chaneels', in_channels)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def flip_h(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim) 
	#print('x_reshaped', x_reshaped)
	y = tf.image.flip_left_right(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])]) 
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def flip_h_py(x):
	return None


# Flip an image vertically (upside down)
def flip_v_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1
	
	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []	
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)
	
	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []	
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)
	
	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	in_channels = remained / in_width
	#print('in_chaneels', in_channels)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def flip_v(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim) 
	#print('x_reshaped', x_reshaped)
	y = tf.image.flip_up_down(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])]) 
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def flip_v_py(x):
	return None


#Converts one or more images from RGB to Grayscale
def rgb_to_g_init(input_size):
	while input_size % 3 != 0:
		input_size -= 1
	
	min_input_size = 6
	if input_size < min_input_size:
		input_size = min_input_size

	in_channels = 3
	remained = input_size / in_channels 

	in_height_candidate = []	
	for i in range(1, remained//2+1):
		if remained % i == 0:
			in_height_candidate.append(i)
	
	in_height = np.random.choice(in_height_candidate)
	in_width = remained / in_height

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def rgb_to_g(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim) 
	#print('x_reshaped', x_reshaped)
	y = tf.image.rgb_to_grayscale(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])]) 
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def rgb_to_g_py(x):
	return None

# Converts one or more images from Grayscale to RGB
def g_to_rgb(x, parameter=None, weight=None):
	x_reshaped = tf.expand_dims(x, -1)
	y = tf.image.grayscale_to_rgb(x_reshaped)
	y_shape = y.get_shape().as_list()
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	return y_reshaped

def g_to_rgb_py(x):
	return None

# Returns vertical image gradients (dy) for each color channel
def img_vgrad_init(input_size):
	while input_size % 3 != 0:
		input_size -= 1

	min_input_size = 12
	if input_size < min_input_size:
		input_size = min_input_size

	in_channels = 3
	remained = input_size / in_channels

	in_height_candidate = []
	for i in range(1, remained//2+1):
		if remained % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	in_width = remained / in_height

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def img_vgrad(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim)
	#print('x_reshaped', x_reshaped)
	y, _ = tf.image.image_gradients(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def img_vgrad_py(x):
	return None

# Returns horizonttal image gradients (dx) for each color channel
def img_hgrad_init(input_size):
	while input_size % 3 != 0:
		input_size -= 1

	min_input_size = 12
	if input_size < min_input_size:
		input_size = min_input_size

	in_channels = 3
	remained = input_size / in_channels

	in_height_candidate = []
	for i in range(1, remained//2+1):
		if remained % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	in_width = remained / in_height

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def img_hgrad(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim)
	#print('x_reshaped', x_reshaped)
	_, y = tf.image.image_gradients(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def img_hgrad_py(x):
	return None

# Returns a tensor holding Sobel edge maps
def sobel_edges_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)

	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	in_channels = remained / in_width

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def sobel_edges(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim)
	#print('x_reshaped', x_reshaped)
	y = tf.image.sobel_edges(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def sobel_edges_py(x):
	return None

# Rotate image counter-clockwise by 90 degrees
def rot90_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)

	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	in_channels = remained / in_width

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def rot90(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim)
	#print('x_reshaped', x_reshaped)
	y = tf.image.rot90(x_reshaped, k=1)
	#y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(input_dim)])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def rot90_py(x):
	return None

# Rotate image counter-clockwise by 180 degrees
def rot180_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)

	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	in_channels = remained / in_width

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def rot180(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim)
	#print('x_reshaped', x_reshaped)
	y = tf.image.rot90(x_reshaped, k=2)
	#y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(input_dim)])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def rot180_py(x):
	return None

# Rotate image counter-clockwise by 270 degrees
def rot270_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)

	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	in_channels = remained / in_width

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def rot270(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim)
	#print('x_reshaped', x_reshaped)
	y = tf.image.rot90(x_reshaped, k=2)
	#y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(input_dim)])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def rot270_py(x):
	return None

# Computes the grayscale dilation
def dilation2d_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)

	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	depth = remained / in_width
	#print('depth', depth)
	filter_height = np.random.randint(low=1, high=in_height)
	#print('filter_height', filter_height)
	filter_width = np.random.randint(low=1, high=in_width)
	#print('filter_width', filter_width)
	stride_height = np.random.randint(low=1, high=in_height-filter_height+1)
	#print('stride_height', stride_height)
	stride_width = np.random.randint(low=1, high=in_width-filter_width+1)
	#print('stride_width', stride_width)

	rate_height = np.random.randint(low=1, high=5)
	rate_width = np.random.randint(low=1, high=5)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, depth ]
	parameter['filter_dim'] = [ filter_height, filter_width, depth ]
	#print('input_dim', parameter['input_dim'])
	#print('filter_dim', parameter['filter_dim'])
	ni = np.prod(parameter['input_dim'])
	no = np.prod(parameter['filter_dim'])

	parameter['strides'] = [ 1, stride_height, stride_width, 1 ]
	#print('strides', parameter['strides'])

	if np.random.randint(2) == 0:
		parameter['padding'] = 'VALID'
	else:
		parameter['padding'] = 'SAME'

	#parameter['rates'] = [ 1, rate_height, rate_width, 1 ]
	parameter['rates'] = [ 1, 1, 1, 1 ]
	#print('rates', parameter['rates'])

	#parameter['weight'] = np.random.random(parameter['filter_dim']).astype('f')*np.sqrt(1/float(ni+no))
	single_weight = np.repeat(np.random.random(1).astype('f'), np.prod(parameter['filter_dim']))
	parameter['weight'] = np.reshape(single_weight, parameter['filter_dim'])

	#print('input_dim', parameter['input_dim'])
	#print('filter_dim ', parameter['filter_dim'])
	#print('strides', parameter['strides'])
	#print('padding', parameter['padding'])
	#print('weight', parameter['weight'])

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def dilation2d(x, parameter=None, weight=None):
	if weight == None:
		weight = parameter['weight']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.nn.dilation2d(x_reshaped, weight, strides=parameter['strides'],
		padding=parameter['padding'], rates=parameter['rates'])
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def dilation2d_py(x):
	return None

# Computes the grayscale erosion
def erosion2d_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_height_candidate = []
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	#print('in_height', in_height)

	remained = input_size / in_height

	in_width_candidate = []
	for i in range(2, remained+1):
		if remained % i == 0:
			in_width_candidate.append(i)

	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)
	depth = remained / in_width
	#print('depth', depth)
	filter_height = np.random.randint(low=1, high=in_height)
	#print('filter_height', filter_height)
	filter_width = np.random.randint(low=1, high=in_width)
	#print('filter_width', filter_width)
	stride_height = np.random.randint(low=1, high=in_height-filter_height+1)
	#print('stride_height', stride_height)
	stride_width = np.random.randint(low=1, high=in_width-filter_width+1)
	#print('stride_width', stride_width)

	rate_height = np.random.randint(low=1, high=5)
	rate_width = np.random.randint(low=1, high=5)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, depth ]
	parameter['filter_dim'] = [ filter_height, filter_width, depth ]
	#print('input_dim', parameter['input_dim'])
	#print('filter_dim', parameter['filter_dim'])
	ni = np.prod(parameter['input_dim'])
	no = np.prod(parameter['filter_dim'])

	parameter['strides'] = [ 1, stride_height, stride_width, 1 ]
	#print('strides', parameter['strides'])

	if np.random.randint(2) == 0:
		parameter['padding'] = 'VALID'
	else:
		parameter['padding'] = 'SAME'

	#parameter['rates'] = [ 1, rate_height, rate_width, 1 ]
	parameter['rates'] = [ 1, 1, 1, 1 ]
	#print('rates', parameter['rates'])

	#parameter['weight'] = np.random.random(parameter['filter_dim']).astype('f')*np.sqrt(1/float(ni+no))
	single_weight = np.repeat(np.random.random(1).astype('f'), np.prod(parameter['filter_dim']))
	parameter['weight'] = np.reshape(single_weight, parameter['filter_dim'])

	#print('input_dim', parameter['input_dim'])
	#print('filter_dim ', parameter['filter_dim'])
	#print('strides', parameter['strides'])
	#print('padding', parameter['padding'])
	#print('weight', parameter['weight'])

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def erosion2d(x, parameter=None, weight=None):
	if weight == None:
		weight = parameter['weight']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.nn.erosion2d(x_reshaped, weight, strides=parameter['strides'],
		padding=parameter['padding'], rates=parameter['rates'])
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def erosion2d_py(x):
	return None


functions = { 
	total_variation.__name__: [ total_variation, total_variation_py, total_variation_init ],
	img_transpose.__name__: [ img_transpose, img_transpose_py, img_transpose_init ],
	flip_h.__name__: [ flip_h, flip_h_py, flip_h_init ],
	flip_v.__name__: [ flip_v, flip_v_py, flip_v_init ],
	rgb_to_g.__name__: [ rgb_to_g, rgb_to_g_py, rgb_to_g_init ],
	g_to_rgb.__name__: [ g_to_rgb, g_to_rgb_py, None ],
	img_vgrad.__name__: [ img_vgrad, img_vgrad_py, img_vgrad_init ],
	img_hgrad.__name__: [ img_hgrad, img_hgrad_py, img_hgrad_init ],
	sobel_edges.__name__: [ sobel_edges, sobel_edges_py, sobel_edges_init ],
	rot90.__name__: [ rot90, rot90_py, rot90_init ],
	rot180.__name__: [ rot180, rot180_py, rot180_init ],
	rot270.__name__: [ rot270, rot270_py, rot270_init ],
	dilation2d.__name__: [ dilation2d, dilation2d, dilation2d_init ],
	erosion2d.__name__: [ erosion2d, erosion2d, erosion2d_init ],
	}


print('%s: %d functions:' % (os.path.splitext(os.path.basename(__file__))[0],
	len(functions)), functions.keys())

#if __name__ == '__main__':
#	main()
