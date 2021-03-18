from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


# Transposes a matrix
def transpose_init(input_size):
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
	in_width = input_size / in_height

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width ] 
	#print('input_dim', parameter['input_dim'])

	return parameter, input_size

def transpose(x, parameter=None, weight=None):
	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.linalg.transpose(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def transpose_py(x, parameter=None, weight=None):
	return None

# Returns the diagonal part of matrix
def diag_part_init(input_size):
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
	in_width = input_size / in_height

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width ] 
	#print('input_dim', parameter['input_dim'])

	return parameter, input_size

def diag_part(x, parameter=None, weight=None):
	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.linalg.diag_part(x_reshaped)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def diag_part_py(x, parameter=None, weight=None):
	return None

# Computes the singular value decompositions (SVD)
def svd_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 4
	max_input_size = 64
	if input_size < min_input_size:
		input_size = min_input_size
	elif input_size > max_input_size:
		input_size = max_input_size

	#print('input_size', input_size)

	in_height_candidate = []
	for i in range(2, input_size//2+1):
		if input_size % i == 0:
			in_height_candidate.append(i)

	in_height = np.random.choice(in_height_candidate)
	in_width = input_size / in_height

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width ] 
	#print('input_dim', parameter['input_dim'])

	return parameter, input_size

def svd(x, parameter=None, weight=None):
	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.linalg.svd(x_reshaped)
	#print('y', y)

	input_dim = parameter['input_dim']
	#print('input_dim', input_dim)

	if input_dim[0] < input_dim[1]:
		small = input_dim[0]
		large = input_dim[1]
	else:
		small = input_dim[1]
		large = input_dim[0]
	output_len = small + input_dim[0]*input_dim[1] + small*small
	#print('output_len', output_len)

	y_list = []
	for y_ in list(y):
		y_flattened = tf.reshape(y_, [-1])
		#print('y_flattened', y_flattened)
		y_list.append(y_flattened)
	
	y_concat = tf.concat(y_list, 0)
	#print('y_concat', y_concat)
	y_concat_shape = y_concat.get_shape().as_list()
	#print('y_concat_shape', y_concat_shape)
	y_reshaped = tf.reshape(y_concat, [-1, output_len])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def svd_py(x, parameter=None, weight=None):
	return None

# Computes the determinant
def det_init(input_size):
	in_width = int(np.sqrt(input_size))

	min_in_width = 2
	max_in_width = 128
	if in_width < min_in_width:
		in_width = min_in_width
	elif in_width > max_in_width:
		in_width = max_in_width

	parameter = {}
	parameter['input_dim'] = [ in_width, in_width ] 
	#print('input_dim', parameter['input_dim'])

	input_size = in_width*in_width
	return parameter, input_size

def det(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']
	#print('input_dim', input_dim)
	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim)
	#print('x_reshaped', x_reshaped)
	y = tf.linalg.det(x_reshaped)
	#print('y', y)
	y_reshaped = tf.reshape(y, [-1, 1])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def det_py(x, parameter=None, weight=None):
	return None

# Compute the trace
def trace_init(input_size):
	in_width = int(np.sqrt(input_size))

	min_in_width = 2
	max_in_width = 128
	if in_width < min_in_width:
		in_width = min_in_width
	elif in_width > max_in_width:
		in_width = max_in_width

	parameter = {}
	parameter['input_dim'] = [ in_width, in_width ] 
	#print('input_dim', parameter['input_dim'])

	input_size = in_width*in_width
	return parameter, input_size

def trace(x, parameter=None, weight=None):
	input_dim = parameter['input_dim']
	#print('input_dim', input_dim)
	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + input_dim)
	#print('x_reshaped', x_reshaped)
	y = tf.linalg.trace(x_reshaped)
	#print('y', y)
	y_reshaped = tf.reshape(y, [-1, 1])
	#print('y_reshaped', y_reshaped)

	return y_reshaped

def trace_py(x, parameter=None, weight=None):
	return None

# Compute the pairwise cross product
def cross_init(input_size):
	while input_size % 6 != 0:
		input_size -= 1

	min_input_size = 6
	if input_size < min_input_size:
		input_size = min_input_size

	in_width = 3
	in_height = input_size / 2 / in_width

	parameter = {}
	parameter['input_size'] = input_size
	parameter['input_dim'] = [ in_height, in_width ]

	return parameter, input_size

def cross(x, parameter=None, weight=None):
	#print('x', x)
	x1 = tf.reshape(x[:, 0:parameter['input_size']//2],
		[-1] + parameter['input_dim'])
	#print('x1', x1)
	x2 = tf.reshape(x[:, parameter['input_size']//2:parameter['input_size']],
		[-1] + parameter['input_dim'])
	#print('x2', x2)
	y = tf.linalg.cross(x1, x2)
	#print('y', y)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])

	return y_reshaped

def cross_py(x, parameter=None, weight=None):
	return None

# Compute the dot product of two vectors
def dot_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 2
	if input_size < min_input_size:
		input_size = min_input_size

	parameter = {}
	parameter['input_size'] = input_size

	return parameter, input_size

def dot(x, parameter=None, weight=None):
	x1 = x[:, 0:parameter['input_size']//2]
	#print('x1', x1)
	x2 = x[:, parameter['input_size']//2:parameter['input_size']]
	#print('x2', x2)
	y = tf.reduce_sum(tf.multiply(x1, x2), axis=1, keep_dims=True)

	return y

def dot_py(x, parameter=None, weight=None):
	return None

# Normalizes tensor along dimension axis using specified norm
def norm(x, parameter=None, weight=None):
	y = tf.linalg.norm(x, axis=1, keep_dims=True)

	return y

def norm_py(x, parameter=None, weight=None):
	return None

"""
# Computes the Cholesky decomposition of one or more square matrices
def cholesky_init(input_size):
	in_height = int(np.floor(np.sqrt(input_size)))
	min_in_height = 4
	if in_height < min_in_height:
		in_height = min_in_height

	in_width = in_height
	input_size = in_height * in_width

	parameter = {}
	parameter['input_dim'] = [ in_height, in_width ]

	return parameter, input_size

def cholesky(x, parameter=None, weight=None):
	#print('x', x)
	x_reshaped = tf.reshape(x, [-1] + parameter['input_dim'])
	#print('x_reshaped', x_reshaped)
	y = tf.linalg.cholesky(x_reshaped)
	#print('y', y)
	y_shape = y.get_shape().as_list()
	#print('y_shape', y_shape)
	y_reshaped = tf.reshape(y, [-1, np.prod(y_shape[1:])])

	return y_reshaped

def cholesky_py(x, parameter=None, weight=None):
	return None
"""

functions = {
	transpose.__name__: [ transpose, transpose_py, transpose_init ],
	diag_part.__name__: [ diag_part, diag_part_py, diag_part_init ],
	#svd.__name__: [ svd, svd_py, svd_init ],
	#det.__name__: [ det, det_py, det_init ],
	trace.__name__: [ trace, trace_py, trace_init ],
	cross.__name__: [ cross, cross_py, cross_init ],
	dot.__name__: [ dot, dot_py, dot_init ],
	norm.__name__: [ norm, norm_py, None ],
	#cholesky.__name__: [ cholesky, cholesky_py, cholesky_init ],
	}


print('%s: %d functions:' % (os.path.splitext(os.path.basename(__file__))[0],
	len(functions)), functions.keys())

#if __name__ == '__main__':
#	main()
