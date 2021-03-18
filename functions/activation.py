from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


# Computes sigmoid
def sigmoid(x, parameter=None, weight=None):
	y = tf.math.sigmoid(x)
	return y

def sigmoid_py(x, parameter=None, weight=None):
	y = np.sigmoid(x).astype(np.float32)
	if not hasattr(y, "__len__"):
		y = [y]
	return y

# Computes rectified linear
def relu(x, parameter=None, weight=None):
	y = tf.nn.relu(x)
	return y

def relu_py(x, parameter=None, weight=None):
	return None

# Compute the Leaky ReLU activation function
def leaky_relu(x, parameter=None, weight=None):
	y = tf.nn.leaky_relu(x)
	return y

def leaky_relu_py(x, parameter=None, weight=None):
	return None

# Computes hyperbolic tangent of x element-wise
def tanh(x, parameter=None, weight=None):
	y = tf.math.tanh(x)
	return y

def tanh_py(x, parameter=None, weight=None):
	return None

# Computes softplus: log(exp(features) + 1)
def softplus(x, parameter=None, weight=None):
	y = tf.math.softplus(x)
	y_no_nan = tf.where(tf.is_nan(y), tf.zeros_like(y), y)
	return y_no_nan

def softplus_py(x, parameter=None, weight=None):
	return None

# Computes softsign: features / (abs(features) + 1)
def softsign(x, parameter=None, weight=None):
	y = tf.math.softsign(x)
	return y

def softsign_py(x, parameter=None, weight=None):
	return None


functions = {
	sigmoid.__name__: [ sigmoid, sigmoid_py, None ],
	relu.__name__: [ relu, relu_py, None ],
	leaky_relu.__name__: [ leaky_relu, leaky_relu_py, None ],
	tanh.__name__: [ tanh, tanh_py, None ],
	softplus.__name__: [ softplus, softplus_py, None ],
	softsign.__name__: [ softplus, softplus_py, None ],
	}


print('%s: %d functions:' % (os.path.splitext(os.path.basename(__file__))[0],
	len(functions)), functions.keys())

#if __name__ == '__main__':
#	main()
