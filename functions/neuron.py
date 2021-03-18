from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


# Fully-connected neuron function
def neuron_init(input_size, output_size=None):
	if output_size is None:
		output_size = input_size

	hidden_layer_size = input_size + output_size

	parameter = {}
	parameter['input_size'] = input_size
	parameter['hidden_layer_size'] = hidden_layer_size
	parameter['output_size'] = output_size
	#print('input_size', parameter['input_size'])
	#print('hidden_layer_size', parameter['hidden_layer_size'])
	#print('output_size', parameter['output_size'])

	weight1 = np.repeat(float(1)/float(input_size),
		input_size*hidden_layer_size).astype('f')
	weight1 = np.reshape(weight1, [input_size, hidden_layer_size])
	#print('weight1', weight1.shape)
	bias1 = np.repeat(0.0000001, hidden_layer_size).astype('f')
	bias1 = np.expand_dims(bias1, axis=0)
	#print('bias1', bias1.shape)

	weight2 = np.repeat(float(1)/float(hidden_layer_size),
		hidden_layer_size*output_size).astype('f')
	weight2 = np.reshape(weight2, [hidden_layer_size, output_size])
	#print('weight2', weight2.shape)
	bias2 = np.repeat(0.0000001, output_size).astype('f')
	bias2 = np.expand_dims(bias2, axis=0)
	#print('bias2', bias2.shape)

	weight1 = weight1.flatten()
	bias1 = bias1.flatten()
	weight2 = weight2.flatten()
	bias2 = bias2.flatten()

	parameter['weight'] = np.concatenate((weight1, bias1, weight2, bias2),
		axis=None)
	#parameter['activation'] = np.random.randint(3)
	return parameter, input_size

def neuron(x, parameter=None, weight=None):
	if weight == None:
		weight = parameter['weight']

	start_idx = 0
	end_idx = parameter['input_size'] * parameter['hidden_layer_size']
	weight1 = weight[start_idx:end_idx]
	weight1 = tf.reshape(weight1, [parameter['input_size'], parameter['hidden_layer_size']])

	start_idx = end_idx
	end_idx += parameter['hidden_layer_size']
	bias1 = weight[start_idx:end_idx]

	start_idx = end_idx
	end_idx += parameter['hidden_layer_size'] * parameter['output_size']
	weight2 = weight[start_idx:end_idx]
	weight2 = tf.reshape(weight2, [parameter['hidden_layer_size'], parameter['output_size']])

	start_idx = end_idx
	end_idx += parameter['output_size']
	bias2 = weight[start_idx:end_idx]

	y1 = tf.add(tf.matmul(x, weight1), bias1)
	y2 = tf.nn.sigmoid(y1)
	y3 = tf.add(tf.matmul(y2, weight2), bias2)

	return y3

def neuron_py(x):
	return None


functions = {
	neuron.__name__: [ neuron, neuron_py, neuron_init ],
	}


print('%s: %d functions:' % (os.path.splitext(os.path.basename(__file__))[0],
	len(functions)), functions.keys())

#if __name__ == '__main__':
#	main()
