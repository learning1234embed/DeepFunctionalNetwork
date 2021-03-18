from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


# Computes sine of x element-wise
def sin(x, parameter=None, weight=None):
	y = tf.math.sin(x)
	return y

def sin_py(x, parameter=None, weight=None):
	return None

# Computes cos of x element-wise
def cos(x, parameter=None, weight=None):
	y = tf.math.cos(x)
	return y

def cos_py(x, parameter=None, weight=None):
	return None

# Computes tan of x element-wise
def tan(x, parameter=None, weight=None):
	y = tf.math.tan(x)
	return y

def tan_py(x, parameter=None, weight=None):
	return None


functions = {
	sin.__name__: [ sin, sin_py, None ],
	cos.__name__: [ cos, cos_py, None ],
	tan.__name__: [ tan, tan_py, None ],
	}


print('%s: %d functions:' % (os.path.splitext(os.path.basename(__file__))[0],
	len(functions)), functions.keys())

#if __name__ == '__main__':
#	main()
