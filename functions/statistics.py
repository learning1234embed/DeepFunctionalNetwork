from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


# Computes the mean of elements
def mean(x, parameter=None, weight=None):
	y = tf.reduce_mean(x, 1, keepdims=True)
	return y

def mean_py(x, parameter=None, weight=None):
	y = np.average(x).astype(np.float32)
	if not hasattr(y, "__len__"):
		y = [y]
	return y

# Variance
def variance(x, parameter=None, weight=None):
	y = tf.math.reduce_variance(x, axis=1, keepdims=True)
	return y

def variance_py(x):
	return None

# Calculate the mean and variance of x
def moments(x, parameter=None, weight=None):
	y = tf.nn.moments(x, axes=1, keep_dims=True)
	y_concat = tf.concat(y, 1)
	return y_concat

def moments_py(x):
	return None

# Standard deviation
def std(x, parameter=None, weight=None):
	y = tf.math.reduce_std(x, axis=1, keepdims=True)
	return y

def std_py(x):
	return None

# Gauss error function
def erf(x, parameter=None, weight=None):
	y = tf.math.erf(x)
	return y

def erf_py(x):
	return None

# Complementary error function
def erfc(x, parameter=None, weight=None):
	y = tf.math.erfc(x)
	return y

def erfc_py(x):
	return None


functions = {
	mean.__name__: [ mean, mean_py, None ],
	variance.__name__: [ variance, variance_py, None ],
	moments.__name__: [ moments, moments_py, None ],
	std.__name__: [ std, std_py, None ],
	erf.__name__: [ erf, erf_py, None ],
	erfc.__name__: [ erfc, erfc_py, None ],
	}


print('%s: %d functions:' % (os.path.splitext(os.path.basename(__file__))[0],
	len(functions)), functions.keys())

#if __name__ == '__main__':
#	main()
