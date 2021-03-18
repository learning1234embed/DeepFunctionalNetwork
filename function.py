from __future__ import print_function
import tensorflow as tf
import numpy as np

import functions.neuron
import functions.signal_proc
import functions.image_proc
import functions.math
import functions.statistics
import functions.linalg
import functions.activation
import functions.trigonometric

class Function:
	def __init__(self, name, input_size, tf_func, py_func=None, init_func=None):
		self.name = name
		self.input_size = input_size
		self.output_size = None
		self.tf_func = tf_func
		self.py_func = py_func
		self.init_func = init_func
		self.parameter = None
		if init_func != None:
			self.parameter, self.input_size = self.init_func(input_size)
		self.tf_test_and_get_output_size()

		"""
		parameter_string = ''
		if self.parameter != None:
			parameter_string += '( '
			for key, value in self.parameter.items():
				parameter_string += key + ': '
				if key == 'weight':
					parameter_string += str(value.shape)
				else:
					parameter_string += str(value)
				parameter_string += ' '
			parameter_string += ')'

		print(self.name, self.input_size, self.output_size, parameter_string)
		"""

	def tf_test_and_get_output_size(self):
		tf.reset_default_graph()
		x = tf.Variable(tf.random_normal([1] + [self.input_size], stddev=0.35))
		y = self.tf_func(x, self.parameter)
		#print('y', y)
		self.output_size = y.get_shape().as_list()[1]
		#print('self.output_size', self.output_size)
		tf.reset_default_graph()

	def tf_exec(self, x, weight=None):
		return self.tf_func(x, self.parameter, weight)

	def py_exec(self, x):
		return self.py_func(x, self.parameter)

class FunctionSet:
	def __init__(self, pool=None):
		if pool is None:
			self.pool = list(functions.math.functions.items()) + \
				list(functions.signal_proc.functions.items()) + \
				list(functions.image_proc.functions.items()) + \
				list(functions.statistics.functions.items()) + \
				list(functions.linalg.functions.items()) + \
				list(functions.activation.functions.items()) + \
				list(functions.trigonometric.functions.items())
		else:
			self.pool = pool

		self.num_of_pool = len(self.pool)
		self.primitive = None # list
		self.prototype = None # dictionary

		print('total %d functions in function pool' % (self.num_of_pool))

	def generate_primitive(self, primitive_idx, use_neuron_function=1):
		self.primitive = [self.pool[idx] for idx in primitive_idx]

		if use_neuron_function == 1:
			for i in range(len(list(functions.neuron.functions.items()))):
				self.primitive.append(list(functions.neuron.functions.items())[i])
		#print(self.primitive)
		#print('len(self.primitive)', len(self.primitive))
