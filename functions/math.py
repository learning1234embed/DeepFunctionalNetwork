from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

# Returns x1 + x2 element-wise
def add_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 2
	if input_size < min_input_size:
		input_size = min_input_size

	parameter = {}
	parameter['input_size'] = input_size

	return parameter, input_size

def add(x, parameter=None, weight=None):
	x1 = x[:, 0:parameter['input_size']//2]
	x2 = x[:, parameter['input_size']//2:parameter['input_size']]
	y = tf.math.add(x1, x2)
	return y

def add_py(x, parameter=None, weight=None):
	return None

# Returns x - y element-wise
def subtract_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 2
	if input_size < min_input_size:
		input_size = min_input_size

	parameter = {}
	parameter['input_size'] = input_size

	return parameter, input_size

def subtract(x, parameter=None, weight=None):
	x1 = x[:, 0:parameter['input_size']//2]
	x2 = x[:, parameter['input_size']//2:parameter['input_size']]
	y = tf.math.subtract(x1, x2)
	return y

def subtract_py(x, parameter=None, weight=None):
	return None

# Returns x * y element-wise
def multiply_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 2
	if input_size < min_input_size:
		input_size = min_input_size

	parameter = {}
	parameter['input_size'] = input_size

	return parameter, input_size

def multiply(x, parameter=None, weight=None):
	x1 = x[:, 0:parameter['input_size']//2]
	x2 = x[:, parameter['input_size']//2:parameter['input_size']]
	y = tf.math.multiply(x1, x2)
	return y

def multiply_py(x, parameter=None, weight=None):
	return None

# Computes Python style division of x by y
def divide_init(input_size):
	if input_size % 2 != 0:
		input_size -= 1

	min_input_size = 2
	if input_size < min_input_size:
		input_size = min_input_size

	parameter = {}
	parameter['input_size'] = input_size

	return parameter, input_size

def divide(x, parameter=None, weight=None):
	x1 = x[:, 0:parameter['input_size']//2]
	x2 = x[:, parameter['input_size']//2:parameter['input_size']]
	y = tf.math.divide(x1, x2)
	y_no_nan = tf.where(tf.is_nan(y), tf.zeros_like(y), y)
	return y_no_nan

def divide_py(x, parameter=None, weight=None):
	return None

# Computes numerical negative value element-wise
def neg(x, parameter=None, weight=None):
	y = tf.math.negative(x)
	return y

def neg_py(x, parameter=None, weight=None):
	return None

# Performs a safe reciprocal operation
def reciproc(x, parameter=None, weight=None):
	y = tf.math.reciprocal(x)
	y_no_nan = tf.where(tf.is_nan(y), tf.zeros_like(y), y)
	return y_no_nan

def reciproc_py(x, parameter=None, weight=None):
	return None

# Computes the sum of elements
def sum(x, parameter=None, weight=None):
	y = tf.reduce_sum(x, axis=1, keepdims=True)
	return y

def sum_py(x, parameter=None, weight=None):
	y = np.sum(x).astype(np.float32)
	if not hasattr(y, "__len__"):
		y = [y]
	return y

# Compute the cumulative sum
def cumsum(x, parameter=None, weight=None):
	y = tf.math.cumsum(x, axis=1)
	return y 

def cumsum_py(x, parameter=None, weight=None):
	y = np.cumsum(x).astype(np.float32)
	if not hasattr(y, "__len__"):
		y = [y]
	return y

# Compute the cumulative product
def cumprod(x, parameter=None, weight=None):
	y = tf.math.cumprod(x, axis=1)
	return y

def cumprod_py(x, parameter=None, weight=None):
	return None

# Computes the maximum of elements
def max(x, parameter=None, weight=None):
	y = tf.math.reduce_max(x, axis=1, keepdims=True)
	return y

def max_py(x, parameter=None, weight=None):
	return None

# Computes the minimum of elements
def min(x, parameter=None, weight=None):
	y = tf.math.reduce_min(x, axis=1, keepdims=True)
	return y

def min_py(x, parameter=None, weight=None):
	return None

# Computes the absolute value of a tensor
def abs(x, parameter=None, weight=None):
	y = tf.math.abs(x)
	return y

def abs_py(x, parameter=None, weight=None):
	return None

# Computes the Bessel i0e function
def bessel_i0e(x, parameter=None, weight=None):
	y = tf.math.bessel_i0e(x)
	return y

def bessel_i0e_py(x, parameter=None, weight=None):
	return None

# Computes the Bessel i1e function
def bessel_i1e(x, parameter=None, weight=None):
	y = tf.math.bessel_i1e(x)
	return y

def bessel_i1e_py(x, parameter=None, weight=None):
	return None

# Returns element-wise smallest integer not less than x
def ceil(x, parameter=None, weight=None):
	y = tf.math.ceil(x)
	return y

def ceil_py(x, parameter=None, weight=None):
	return None

# Returns element-wise largest integer not greater than x
def floor(x, parameter=None, weight=None):
	y = tf.math.floor(x)
	return y

def floor_py(x, parameter=None, weight=None):
	return None

# Rounds the values of a tensor to the nearest integer, element-wise
def round(x, parameter=None, weight=None):
	y = tf.math.floor(x)
	return y

def round_py(x, parameter=None, weight=None):
	return None

# Computes square root of x element-wise
def sqrt(x, parameter=None, weight=None):
	y = tf.math.sqrt(x)
	y_no_nan = tf.where(tf.is_nan(y), tf.zeros_like(y), y)
	return y_no_nan

def sqrt_py(x, parameter=None, weight=None):
	return None

# Computes square of x element-wise
def square(x, parameter=None, weight=None):
	y = tf.math.square(x)
	return y

def square_py(x, parameter=None, weight=None):
	return None

# Computes Psi, the derivative of Lgamma (the log of the absolute value of Gamma(x))
def digamma(x, parameter=None, weight=None):
	y = tf.math.digamma(x)
	y_no_nan = tf.where(tf.is_nan(y), tf.zeros_like(y), y)
	return y_no_nan

def digamma_py(x, parameter=None, weight=None):
	return None

# Computes the log of the absolute value of Gamma(x)
def lgamma(x, parameter=None, weight=None):
	y = tf.math.lgamma(x)
	return y

def lgamma_py(x, parameter=None, weight=None):
	return None

# Compute ln befa function
def lbeta(x, parameter=None, weight=None):
	y = tf.math.lbeta(x)
	y_shape = y.get_shape().as_list()
	y_reshaped = tf.reshape(y, [-1, 1])

	return y_reshaped

def lbeta_py(x, parameter=None, weight=None):
	return None


functions = {
	add.__name__: [ add, add_py, add_init ],
	subtract.__name__: [ subtract, subtract_py, subtract_init ],
	multiply.__name__: [ multiply, multiply_py, multiply_init ],
	divide.__name__: [ divide, divide_py, divide_init ],
	sum.__name__: [ sum, sum_py, None ],
	neg.__name__: [ neg, neg_py, None ],
	reciproc.__name__: [ reciproc, reciproc_py, None ],
	cumsum.__name__: [ cumsum, cumsum_py, None ],
	cumprod.__name__: [ cumprod, cumprod_py, None ],
	max.__name__: [ max, max_py, None ],
	min.__name__: [ min, min_py, None ],
	abs.__name__: [ abs, abs_py, None ],
	bessel_i0e.__name__: [ bessel_i0e, bessel_i0e_py, None ],
	bessel_i1e.__name__: [ bessel_i1e, bessel_i1e_py, None ],
	ceil.__name__: [ ceil, ceil_py, None ],
	floor.__name__: [ floor, floor_py, None ],
	round.__name__: [ round, round_py, None ],
	sqrt.__name__: [ sqrt, sqrt_py, None ],
	square.__name__: [ square, square_py, None ],
	lgamma.__name__: [ lgamma, lgamma_py, None ],
	digamma.__name__: [ digamma, digamma_py, None ],
	lbeta.__name__: [ lbeta, lbeta_py, None ],
	}


print('%s: %d functions:' % (os.path.splitext(os.path.basename(__file__))[0],
	len(functions)), functions.keys())

#if __name__ == '__main__':
#	main()
