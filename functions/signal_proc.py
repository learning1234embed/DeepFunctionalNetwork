from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


# Computes the 1D Discrete Cosine Transform (DCT)
def dct(x, parameter=None, weight=None):
	y = tf.signal.dct(x)
	return y

def dct_py(x, parameter=None, weight=None):
	return None

# Computes the 1D Inverse Discrete Cosine Transform (DCT) of input
def idct(x, parameter=None, weight=None):
	y = tf.signal.idct(x)
	return y

def idct_py(x, parameter=None, weight=None):
	return None

# Computes the 1D Inverse Discrete Cosine Transform (DCT)
def idct(x, parameter=None, weight=None):
	y = tf.signal.idct(x)
	return y

def idct_py(x, parameter=None, weight=None):
	return None

# Real-valued fast Fourier transform.
def rfft(x, parameter=None, weight=None):
	y = tf.math.real(tf.signal.rfft(x))
	return y

def rfft_py(x, parameter=None, weight=None):
	y = np.real(np.fft.rfft(x)).astype(np.float32)
	if not hasattr(y, "__len__"):
		y = [y]
	return y

# 2D real-valued fast Fourier transform
def rfft2d_init(input_size):
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

	in_width = input_size / in_height
	#print('in_width', in_width)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width ]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def rfft2d(x, parameter=None, weight=None):
	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.math.real(tf.signal.rfft2d(x_reshaped))
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])

	return y_reshaped

def rfft2d_py(x, parameter=None, weight=None):
	return None

# Inverse real-valued fast Fourier transform
def irfft(x, parameter=None, weight=None):
	x_cast = tf.cast(x, tf.complex64)
	y = tf.signal.irfft(x_cast)
	return y

def irfft_py(x, parameter=None, weight=None):
	return None

# Inverse 2D fast Fourier transform
def irfft2d_init(input_size):
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

	in_width = input_size / in_height
	#print('in_width', in_width)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width ]

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def irfft2d(x, parameter=None, weight=None):
	#print('x', x)
	x_cast = tf.cast(x, tf.complex64)
	x_reshaped = tf.reshape(x_cast, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.math.real(tf.signal.irfft2d(x_reshaped))
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])

	return y_reshaped

def irfft2d_py(x, parameter=None, weight=None):
	return None

# Computes the Short-time Fourier Transform
def stft_init(input_size):
	min_input_size = 16
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	parameter = {}
	parameter['frame_length'] = np.random.randint(low=2, high=(input_size+1)//2)
	#print('frame_length', parameter['frame_length'])
	parameter['frame_step'] = np.random.randint(low=1, high=parameter['frame_length'])
	#print('frame_step', parameter['frame_step'])

	return parameter, input_size

def stft(x, parameter=None, weight=None):
	y = tf.math.real(tf.signal.stft(x, parameter['frame_length'], parameter['frame_step']))
	#print('y', y)
	y_shape = y.get_shape().as_list()
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#y_reshaped = tf.reshape(y, [-1, tf.size(y)])
	#print('y_reshaped', y_reshaped)
	return y_reshaped

def stft_py(x, parameter=None, weight=None):
	return None

# Computes a 1-D convolution
def conv1d_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 4
	if input_size < min_input_size:
		input_size = min_input_size

	#print('input_size', input_size)

	in_width_candidate = []
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_width_candidate.append(i)

	in_width = np.random.choice(in_width_candidate)
	#print('in_width', in_width)

	in_channels = input_size / in_width
	#print('in_channels', in_channels)
	out_channels = np.random.randint(low=1, high=33)
	#print('out_channels', out_channels)


	filter_width = np.random.randint(low=1, high=in_width)
	#print('filter_width', filter_width)
	stride_width = np.random.randint(low=1, high=in_width-filter_width+1)
	#print('stride_width', stride_width)

	parameter = {}
	parameter['input_dim'] = [ in_width, in_channels]
	parameter['conv_dim'] = [ filter_width, in_channels, out_channels ]
	ni = np.prod(parameter['input_dim'])
	no = np.prod(parameter['conv_dim'])

	parameter['strides'] = stride_width

	if np.random.randint(2) == 0:
		parameter['padding'] = 'VALID'
	else:
		parameter['padding'] = 'SAME'

	#parameter['weight'] = np.random.random(parameter['conv_dim']).astype('f')*np.sqrt(1/float(ni+no))
	single_weight = np.repeat(np.random.random(1).astype('f'), np.prod(parameter['conv_dim']))
	parameter['weight'] = np.reshape(single_weight, parameter['conv_dim'])

	#print('input_dim', parameter['input_dim'])
	#print('conv_dim ', parameter['conv_dim'])
	#print('strides', parameter['strides'])
	#print('padding', parameter['padding'])
	#print('weight', parameter['weight'])

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def conv1d(x, parameter=None, weight=None):
	if weight == None:
		weight = parameter['weight']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.nn.conv1d(x_reshaped, weight, stride=parameter['strides'],
		padding=parameter['padding'])
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def conv1d_py(x):
	return None

# Computes a 2-D convolution
def conv2d_init(input_size):
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
	#print('in_channels', in_channels)
	out_channels = np.random.randint(low=1, high=33)
	#print('out_channels', out_channels)
	filter_height = np.random.randint(low=1, high=in_height)
	#print('filter_height', filter_height)
	filter_width = np.random.randint(low=1, high=in_width)
	#print('filter_width', filter_width)
	stride_height = np.random.randint(low=1, high=in_height-filter_height+1)
	#print('stride_height', stride_height)
	stride_width = np.random.randint(low=1, high=in_width-filter_width+1)
	#print('stride_width', stride_width)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]
	parameter['conv_dim'] = [ filter_height, filter_width, in_channels, out_channels ]
	ni = np.prod(parameter['input_dim'])
	no = np.prod(parameter['conv_dim'])

	parameter['strides'] = [ 1, stride_height, stride_width, 1 ]

	if np.random.randint(2) == 0:
		parameter['padding'] = 'VALID'
	else:
		parameter['padding'] = 'SAME'

	#parameter['weight'] = np.random.random(parameter['conv_dim']).astype('f')*np.sqrt(1/float(ni+no))
	single_weight = np.repeat(np.random.random(1).astype('f'), np.prod(parameter['conv_dim']))
	parameter['weight'] = np.reshape(single_weight, parameter['conv_dim'])

	#print('input_dim', parameter['input_dim'])
	#print('conv_dim ', parameter['conv_dim'])
	#print('strides', parameter['strides'])
	#print('padding', parameter['padding'])
	#print('weight', parameter['weight'])

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def conv2d(x, parameter=None, weight=None):
	if weight == None:
		weight = parameter['weight']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.nn.conv2d(x_reshaped, weight, strides=parameter['strides'],
		padding=parameter['padding'])
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def conv2d_py(x):
	return None

# Depthwise 2-D convolution
def depthwise_conv2d_init(input_size):
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
	#print('in_channels', in_channels)
	channel_multiplier = np.random.randint(low=1, high=16)
	#print('channel_multiplier', channel_multiplier)
	filter_height = np.random.randint(low=1, high=in_height)
	#print('filter_height', filter_height)
	filter_width = np.random.randint(low=1, high=in_width)
	#print('filter_width', filter_width)
	stride_height = np.random.randint(low=1, high=in_height-filter_height+1)
	#print('stride_height', stride_height)
	stride_width = np.random.randint(low=1, high=in_width-filter_width+1)
	#print('stride_width', stride_width)

	if stride_height < stride_width:
		min_stride = stride_height
	else:
		min_stride = stride_width

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]
	parameter['conv_dim'] = [ filter_height, filter_width, in_channels, channel_multiplier ]
	ni = np.prod(parameter['input_dim'])
	no = np.prod(parameter['conv_dim'])

	parameter['strides'] = [ 1, min_stride, min_stride, 1 ]

	if np.random.randint(2) == 0:
		parameter['padding'] = 'VALID'
	else:
		parameter['padding'] = 'SAME'

	#parameter['weight'] = np.random.random(parameter['conv_dim']).astype('f')*np.sqrt(1/float(ni+no))
	single_weight = np.repeat(np.random.random(1).astype('f'), np.prod(parameter['conv_dim']))
	parameter['weight'] = np.reshape(single_weight, parameter['conv_dim'])

	#print('input_dim', parameter['input_dim'])
	#print('conv_dim ', parameter['conv_dim'])
	#print('strides', parameter['strides'])
	#print('padding', parameter['padding'])
	#print('weight', parameter['weight'])

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def depthwise_conv2d(x, parameter=None, weight=None):
	if weight == None:
		weight = parameter['weight']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.nn.depthwise_conv2d(x_reshaped, weight, strides=parameter['strides'],
		padding=parameter['padding'])
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def depthwise_conv2d_py(x):
	return None

# Performs the max pooling on the input
def max_pool_init(input_size):
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
	#print('in_channels', in_channels)
	out_channels = np.random.randint(low=1, high=33)
	#print('out_channels', out_channels)
	filter_height = np.random.randint(low=1, high=in_height)
	#print('filter_height', filter_height)
	filter_width = np.random.randint(low=1, high=in_width)
	#print('filter_width', filter_width)
	stride_height = np.random.randint(low=1, high=in_height-filter_height+1)
	#print('stride_height', stride_height)
	stride_width = np.random.randint(low=1, high=in_width-filter_width+1)
	#print('stride_width', stride_width)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]
	parameter['conv_dim'] = [ filter_height, filter_width, in_channels, out_channels ]
	ni = np.prod(parameter['input_dim'])
	no = np.prod(parameter['conv_dim'])

	parameter['strides'] = [ 1, stride_height, stride_width, 1 ]
	parameter['ksize'] = [ 1, stride_height, stride_width, 1 ]

	if np.random.randint(2) == 0:
		parameter['padding'] = 'VALID'
	else:
		parameter['padding'] = 'SAME'

	#parameter['weight'] = np.random.random(parameter['conv_dim']).astype('f')*np.sqrt(1/float(ni+no))
	single_weight = np.repeat(np.random.random(1).astype('f'), np.prod(parameter['conv_dim']))
	parameter['weight'] = np.reshape(single_weight, parameter['conv_dim'])

	#print('input_dim', parameter['input_dim'])
	#print('conv_dim ', parameter['conv_dim'])
	#print('strides', parameter['strides'])
	#print('padding', parameter['padding'])
	#print('weight', parameter['weight'])

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def max_pool(x, parameter=None, weight=None):
	if weight == None:
		weight = parameter['weight']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.nn.max_pool(x_reshaped, ksize=parameter['ksize'],
		strides=parameter['strides'], padding=parameter['padding'])
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def max_pool_py(x):
	return None

# Performs the average pooling on the input
def avg_pool_init(input_size):
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
	#print('in_channels', in_channels)
	out_channels = np.random.randint(low=1, high=33)
	#print('out_channels', out_channels)
	filter_height = np.random.randint(low=1, high=in_height)
	#print('filter_height', filter_height)
	filter_width = np.random.randint(low=1, high=in_width)
	#print('filter_width', filter_width)
	stride_height = np.random.randint(low=1, high=in_height-filter_height+1)
	#print('stride_height', stride_height)
	stride_width = np.random.randint(low=1, high=in_width-filter_width+1)
	#print('stride_width', stride_width)

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width, in_channels]
	parameter['conv_dim'] = [ filter_height, filter_width, in_channels, out_channels ]
	ni = np.prod(parameter['input_dim'])
	no = np.prod(parameter['conv_dim'])

	parameter['strides'] = [ 1, stride_height, stride_width, 1 ]
	parameter['ksize'] = [ 1, stride_height, stride_width, 1 ]

	if np.random.randint(2) == 0:
		parameter['padding'] = 'VALID'
	else:
		parameter['padding'] = 'SAME'

	#parameter['weight'] = np.random.random(parameter['conv_dim']).astype('f')*np.sqrt(1/float(ni+no))
	single_weight = np.repeat(np.random.random(1).astype('f'), np.prod(parameter['conv_dim']))
	parameter['weight'] = np.reshape(single_weight, parameter['conv_dim'])

	#print('input_dim', parameter['input_dim'])
	#print('conv_dim ', parameter['conv_dim'])
	#print('strides', parameter['strides'])
	#print('padding', parameter['padding'])
	#print('weight', parameter['weight'])

	assert np.prod(parameter['input_dim']) == input_size

	return parameter, input_size

def avg_pool(x, parameter=None, weight=None):
	if weight == None:
		weight = parameter['weight']

	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.nn.avg_pool(x_reshaped, ksize=parameter['ksize'],
		strides=parameter['strides'], padding=parameter['padding'])
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def avg_pool_py(x):
	return None


functions = {
	dct.__name__: [ dct, dct_py, None ],
	idct.__name__: [ idct, idct_py, None ],
	rfft.__name__: [ rfft, rfft_py, None ],
	rfft2d.__name__: [ rfft2d, rfft2d_py, rfft2d_init ],
	irfft.__name__: [ irfft, irfft_py, None ],
	irfft2d.__name__: [ irfft2d, irfft2d_py, irfft2d_init ],
	stft.__name__: [ stft, stft_py, stft_init ],
	conv1d.__name__: [ conv1d, conv1d_py, conv1d_init ],
	conv2d.__name__: [ conv2d, conv2d_py, conv2d_init ],
	depthwise_conv2d.__name__: [ depthwise_conv2d, depthwise_conv2d_py, depthwise_conv2d_init ],
	max_pool.__name__: [ max_pool, max_pool_py, max_pool_init ],
	avg_pool.__name__: [ avg_pool, avg_pool_py, avg_pool_init ],
	}


print('%s: %d functions:' % (os.path.splitext(os.path.basename(__file__))[0],
	len(functions)), functions.keys())

#if __name__ == '__main__':
#	main()
