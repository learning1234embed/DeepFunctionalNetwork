from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import copy
import pickle 
import sys
import argparse
import importlib
import shutil
import time

from dfn import DFN
from function import FunctionSet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
#np.set_printoptions(threshold=np.nan)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

class FunctionEstimator:
	def __init__(self, dnn_name, func_set, layers_str, input_size, output_size,
		placeholder_size, fe_dir=None):
		self.dnn_name = dnn_name
		self.func_set = func_set
		self.layers = self.parse_layers(layers_str)
		self.layer_type, self.num_of_neuron_per_layer, self.num_of_weight_per_layer,\
			self.num_of_bias_per_layer = self.calculate_num_of_weight(self.layers)

		self.num_of_neuron = 0
		for layer in self.num_of_neuron_per_layer:
			self.num_of_neuron += np.prod(layer)
			
		self.num_of_weight = sum(self.num_of_weight_per_layer)
		self.num_of_bias = sum(self.num_of_bias_per_layer)

		self.fe_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
			dnn_name), os.path.splitext(os.path.basename(__file__))[0])
		self.fe_file_name = os.path.splitext(os.path.basename(__file__))[0] 
		self.fe_file_path = os.path.join(self.fe_dir, self.fe_file_name)
		self.distribution_file_name = 'function_distribution.npy'
		self.distribution_file_path = os.path.join(self.fe_dir, self.distribution_file_name)

		self.train_data_name = 'fe_train_data.npy'
		self.train_data_path = os.path.join(self.fe_dir, self.train_data_name)
		self.train_label_name = 'fe_train_label.npy'
		self.train_label_path = os.path.join(self.fe_dir, self.train_label_name)
		self.test_data_name = 'fe_test_data.npy'
		self.test_data_path = os.path.join(self.fe_dir, self.test_data_name)
		self.test_label_name = 'fe_test_label.npy'
		self.test_label_path = os.path.join(self.fe_dir, self.test_label_name)
		self.val_data_name = 'fe_val_data.npy'
		self.val_data_path = os.path.join(self.fe_dir, self.val_data_name)
		self.val_label_name = 'fe_val_label.npy'
		self.val_label_path = os.path.join(self.fe_dir, self.val_label_name)
		self.tf_batch_size = 3000

		self.neuron_base_name = "neuron_"
		self.weight_base_name = "weight_"
		self.bias_base_name = "bias_"
		
		self.input_size = input_size
		self.output_size = output_size
		self.placeholder_size = placeholder_size

		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.buildNetwork(sess)
				if not os.path.exists(self.fe_dir):
					os.makedirs(self.fe_dir)
				self.saveNetwork(sess)

	def generate_random_data(self, func_set, program_size, input_data_size, input_low, input_high):
		print('generating %d program with input size %d and placholder size %d' % (program_size,
			input_data_size, self.placeholder_size))

		function_pool = self.func_set.pool
		data_list = []
		label_list = []

		for i in range(program_size):
			primitive_idx = range(len(function_pool))
			self.func_set.generate_primitive(primitive_idx, use_neuron_function=0)
			fd = DFN(self.dnn_name, self.input_size, self.output_size,
				1, self.placeholder_size, primitive_function=self.func_set.primitive,
				use_neuron_function=0)
			fd.generate_population(1)
			for _ in range(np.random.randint(8)+1):
				fd.mutate_individual(fd.population[0])

			function_order = fd.population[0].genotype[0]
			valid_function = fd.population[0].valid_function

			valid_function_name = []
			for j in range(len(function_order)):
				func_name = function_order[j].split(":",1)[0]
				if valid_function[j] == 1:
					valid_function_name.append(func_name)

			label_idx = []
			for j in range(len(self.func_set.primitive)):
				for func_name in valid_function_name:
					if func_name == self.func_set.primitive[j][0]:
						label_idx.append(j)
						break

			occurrence_frequency = [0.0] * len(self.func_set.primitive)
			for j in range(len(self.func_set.primitive)):
				for func_name in valid_function_name:
					if func_name == self.func_set.primitive[j][0]:
						occurrence_frequency[j] += 1
			occurrence_frequency_sum = float(np.sum(occurrence_frequency))
			if occurrence_frequency_sum > 0:
				occurrence_frequency_norm = np.asarray(occurrence_frequency) / occurrence_frequency_sum
			else:
				occurrence_frequency_norm = occurrence_frequency

			input_data = np.random.uniform(low=input_low, high=input_high,
				size=(input_data_size,self.input_size))

			output_data = fd.execute_individual_tf(fd.population[0], input_data)

			data = np.concatenate([input_data, output_data], axis=1)
			data_list.append(data)

			label = np.zeros((input_data_size, len(function_pool)))
			for idx in label_idx:
				label[np.arange(label.shape[0]), idx] = occurrence_frequency_norm[idx]

			label_list.append(label)

			if i % 10 == 0 or i == program_size-1:
				print(i)

		dataset = np.concatenate(data_list)
		label_set = np.concatenate(label_list, axis=0)
		np.random.shuffle(dataset)
		np.random.shuffle(label_set)
		return dataset, label_set

	def generate_training_data(self, num_of_program=1000, input_data_size=1000,
		input_low=0.0, input_high=1.0):
		print('total %d functions in function pool' % len(self.func_set.pool))

		# generate data of [num_of_program*input_data_size, input_size+output+size]
		# generate label of [num_of_program*input_data_size, len(self.func_set.pool)]
		train_data, train_label = self.generate_random_data(self.func_set.pool,
			num_of_program, input_data_size, input_low, input_high)
		print('train_data', train_data.shape)
		print('train_label', train_label.shape)
		np.save(self.train_data_path, train_data)
		np.save(self.train_label_path, train_label)

		test_data, test_label = self.generate_random_data(self.func_set.pool,
			num_of_program//10, input_data_size//10, input_low, input_high)
		print('test_data', test_data.shape)
		print('test_label', test_label.shape)
		np.save(self.test_data_path, test_data)
		np.save(self.test_label_path, test_label)

		val_data, val_label = self.generate_random_data(self.func_set.pool,
			num_of_program//10, input_data_size//10, input_low, input_high)
		print('val_data', val_data.shape)
		print('val_label', val_label.shape)
		np.save(self.val_data_path, val_data)
		np.save(self.val_label_path, val_label)

	def buildNetwork(self, sess):
		layer_type = copy.deepcopy(self.layer_type)
		layer_type = list(filter(lambda type: type != 'max_pool', layer_type))
		layers = self.layers
		parameters = {}
		neurons = {}
		parameters_to_regularize = []
		input_emb_size = 512
		output_emb_size = 512

		keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		original_input = tf.placeholder(tf.float32, [None]+layers[0],
			name="neuron_-1")
		flattened = tf.reshape(original_input, [-1, self.input_size+self.output_size])
		input_part = flattened[:,0:self.input_size]
		output_part = flattened[:,self.input_size:self.input_size+self.output_size]
		iw = tf.get_variable('input_part_weight', shape=([self.input_size,
			input_emb_size]),
			initializer=tf.contrib.layers.xavier_initializer())
		ib = tf.get_variable('input_part_bias', shape=(input_emb_size),
			initializer=tf.contrib.layers.xavier_initializer())
		input_emb = tf.add(tf.matmul(input_part, iw), ib)
		print('input_emb', input_emb)

		ow = tf.get_variable('output_part_weight', shape=([self.output_size,
			output_emb_size]),
			initializer=tf.contrib.layers.xavier_initializer())
		ob = tf.get_variable('output_part_bias', shape=(output_emb_size),
			initializer=tf.contrib.layers.xavier_initializer())
		output_emb = tf.add(tf.matmul(output_part, ow), ob)
		print('output_emb', output_emb)

		#neurons[0] = tf.nn.leaky_relu(tf.concat([input_emb, output_emb], axis=1),
		#	name=self.neuron_base_name+'0')
		neurons[0] = tf.nn.sigmoid(tf.concat([input_emb, output_emb], axis=1),
			name=self.neuron_base_name+'0')

		#neurons[0] = tf.placeholder(tf.float32, [None]+layers[0],
		#	name=self.neuron_base_name+'0')
		print(neurons[0])

		for layer_no in range(1, len(layers)):
			weight_name = self.weight_base_name + str(layer_no-1)
			bias_name = self.bias_base_name + str(layer_no-1)
			neuron_name = self.neuron_base_name + str(layer_no)

			print('self.num_of_neuron_per_layer[layer_no]', self.num_of_neuron_per_layer[layer_no])
	
			if layer_type[layer_no] == "conv":
				conv_parameter = {
					'weights': tf.get_variable(weight_name,
						shape=(layers[layer_no]),
						initializer=tf.contrib.layers.xavier_initializer()),
					'biases' : tf.get_variable(bias_name,
						shape=(layers[layer_no][3]),
						initializer=tf.contrib.layers.xavier_initializer()),
				}

				#parameters_to_regularize.append(tf.reshape(conv_parameter['weights'],
					#[tf.size(conv_parameter['weights'])]))
				#parameters_to_regularize.append(tf.reshape(conv_parameter['biases'],
					#[tf.size(conv_parameter['biases'])]))

				parameters[layer_no-1] = conv_parameter
				print('conv_parameter', parameters[layer_no-1])

				rank = sess.run(tf.rank(neurons[layer_no-1]))

				for _ in range(4 - rank):
					neurons[layer_no-1] = tf.expand_dims(neurons[layer_no-1], -1)

				# CNN
				strides = 1
				output = tf.nn.conv2d(neurons[layer_no-1],
					conv_parameter['weights'],
					strides=[1, strides, strides, 1], padding='VALID')
				output_biased = tf.nn.bias_add(output, conv_parameter['biases'])

				# max pooling
				k = 2
				#neuron = tf.nn.max_pool(tf.nn.leaky_relu(output_biased),
				neuron = tf.nn.max_pool(tf.nn.sigmoid(output_biased),
					ksize=[1, k, k, 1],
					strides=[1, k, k, 1], padding='VALID', name=neuron_name)
				neurons[layer_no] = neuron

			elif layer_type[layer_no] == "hidden" or layer_type[layer_no] == "output":
				fc_parameter = {
					'weights': tf.get_variable(weight_name,
						#shape=(np.prod(self.num_of_neuron_per_layer[layer_no-1]),
						shape=(neurons[layer_no-1].get_shape().as_list()[1],
                	                        np.prod(self.num_of_neuron_per_layer[layer_no])),
                        	                initializer=tf.contrib.layers.xavier_initializer()), 
					'biases' : tf.get_variable(bias_name,
						shape=(np.prod(self.num_of_neuron_per_layer[layer_no])),
						initializer=tf.contrib.layers.xavier_initializer()),
				}

				parameters_to_regularize.append(tf.reshape(fc_parameter['weights'],
					[tf.size(fc_parameter['weights'])]))
				parameters_to_regularize.append(tf.reshape(fc_parameter['biases'],
					[tf.size(fc_parameter['biases'])]))

				parameters[layer_no-1] = fc_parameter
				print('fc_parameter', parameters[layer_no-1])

				# fully-connected
				flattened = tf.reshape(neurons[layer_no-1],
					#[-1, np.prod(self.num_of_neuron_per_layer[layer_no-1])])
					[-1, neurons[layer_no-1].get_shape().as_list()[1]])
				neuron_drop = tf.nn.dropout(flattened, rate=1 - keep_prob)

				if layer_type[layer_no] == "hidden":
					#neuron = tf.nn.leaky_relu(tf.add(tf.matmul(neuron_drop,
					#	fc_parameter['weights']), fc_parameter['biases']),
					#	name=neuron_name)
					neuron = tf.nn.sigmoid(tf.add(tf.matmul(neuron_drop,
						fc_parameter['weights']), fc_parameter['biases']),
						name=neuron_name)

				elif layer_type[layer_no] == "output":
					#y_b = tf.add(tf.matmul(neuron_drop, fc_parameter['weights']),
					#	fc_parameter['biases'])
					#neuron = tf.divide(tf.exp(y_b-tf.reduce_max(y_b)),
					#	tf.reduce_sum(tf.exp(y_b-tf.reduce_max(y_b))),
					#	name=neuron_name)
					output = tf.add(tf.matmul(neuron_drop,
						fc_parameter['weights']), fc_parameter['biases'], name='output')
					print(output)
					#neuron = tf.nn.softmax(output, name=neuron_name)
					neuron = tf.nn.sigmoid(output, name=neuron_name)

				neurons[layer_no] = neuron
			print(neuron)

		# input
		x = neurons[0]

		# output
		y = neurons[len(layers)-1]

		# correct labels
		y_ = tf.placeholder(tf.float32, [None] + layers[-1], name='y_')

		# define the loss function
		regularization = 0.000001 * tf.nn.l2_loss(tf.concat(parameters_to_regularize, 0))
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y) + (1-y_)*tf.log(1-y),
			reduction_indices=[1]), name='cross_entropy') + regularization
		mse = tf.reduce_mean(tf.square(y_ - y), name='mse') + regularization

		# define diff
		#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1),
		#	name='correct_prediction')
		#correct_prediction = tf.equal(tf.round(y), y_,
		#	name='correct_prediction')
		#diff = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
		#	name='diff')
		diff = tf.reduce_mean(tf.abs(y_ - y), name='diff')

		# for training
		learning_rate = 0.001
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
			#name='optimizer').minimize(mse)
			name='optimizer').minimize(cross_entropy)

		init = tf.global_variables_initializer()
		sess.run(init)
	
	def loadNetwork(self, sess):
		saver = tf.train.import_meta_graph(self.fe_file_path + '.meta')
		saver.restore(sess, self.fe_file_path)

	def saveNetwork(self, sess):
		saver = tf.train.Saver()
		saver.save(sess, self.fe_file_path)

	def doTrain(self, sess, graph, train_set, validation_set, batch_size,
		train_iteration, optimizer):
		print("doTrain")

		# get tensors
		tensor_x_name = "neuron_0:0"
		x = graph.get_tensor_by_name("neuron_-1:0")
		y_ = graph.get_tensor_by_name("y_:0")
		keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
		keep_prob = graph.get_tensor_by_name("keep_prob:0")
		diff = graph.get_tensor_by_name("diff:0")
		cross_entropy = graph.get_tensor_by_name("cross_entropy:0")
		mse = graph.get_tensor_by_name("mse:0")

		input_images_validation = validation_set[0]
		input_images_validation_reshaped = np.reshape(validation_set[0],
			([-1] + x.get_shape().as_list()[1:]))
		labels_validation = validation_set[1]

		lowest_diff = None

		# train
		for i in range(train_iteration):
			input_data, labels = self.next_batch(train_set, batch_size)
			input_data_reshpaed = \
				np.reshape(input_data, ([-1] + x.get_shape().as_list()[1:]))

			if i % (100) == 0 or i == (train_iteration-1):
				train_diff, ce, m = sess.run([diff, cross_entropy, mse],
					feed_dict={x: input_data_reshpaed,
					y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})
				print("step %d, training diff: %f ce: %f mse: %f" % (i, train_diff, ce, m))
			
				# validate
				test_diff, ce, m = sess.run([diff, cross_entropy, mse], feed_dict={
					x: input_images_validation_reshaped, y_: labels_validation,
					keep_prob_input: 1.0, keep_prob: 1.0})
				print("step %d, Validation diff: %f ce: %f mse: %f" % (i, test_diff, ce, m))

				if i == 0:
					lowest_diff = test_diff
				else:
					if test_diff < lowest_diff:
						self.saveNetwork(sess)
						lowest_diff = test_diff
						#print('saveNetwork for', lowest_diff)

			sess.run(optimizer, feed_dict={x: input_data_reshpaed,
				y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})

	def train(self, train_set, validation_set, batch_size, train_iteration):
		print("train")
		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.loadNetwork(sess)
				optimizer = graph.get_operation_by_name("optimizer")
				self.doTrain(sess, graph, train_set, validation_set, batch_size,
					train_iteration, optimizer)

	def doInfer(self, sess, graph, data_set, label=None):
		tensor_x_name = "neuron_-1:0"
		x = graph.get_tensor_by_name(tensor_x_name)
		tensor_y_name = "neuron_" + str(len(self.layers)-1) + ":0"
		y = graph.get_tensor_by_name(tensor_y_name)
		keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
		keep_prob = graph.get_tensor_by_name("keep_prob:0")

		# infer
		data_set_reshaped = np.reshape(data_set, ([-1] + x.get_shape().as_list()[1:]))

		iteration = data_set_reshaped.shape[0] // self.tf_batch_size
		remained = data_set_reshaped.shape[0] % self.tf_batch_size

		infer_result_list = []
		for i in range(iteration):
			infer_result = sess.run(y, feed_dict={
				x: data_set_reshaped[i*self.tf_batch_size:i*self.tf_batch_size+self.tf_batch_size],
				keep_prob_input: 1.0, keep_prob: 1.0})
			infer_result_list.append(infer_result)

		if remained > 0:
			infer_result = sess.run(y, feed_dict={
				x: data_set_reshaped[iteration*self.tf_batch_size:iteration*self.tf_batch_size+remained],
				keep_prob_input: 1.0, keep_prob: 1.0})
			infer_result_list.append(infer_result)

		infer_result_set = np.vstack(infer_result_list)

		if label is not None:
			# validate (this is for test)
			y_ = graph.get_tensor_by_name("y_:0")
			diff = graph.get_tensor_by_name("diff:0")
			test_diff = sess.run(diff, feed_dict={
				x: data_set_reshaped, y_: label, keep_prob_input: 1.0,
				keep_prob: 1.0})
			print("Inference diff: %f" % test_diff)

		return infer_result_set

	def infer(self, data_set, label=None):
		print("infer")
		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.loadNetwork(sess)
				return self.doInfer(sess, graph, data_set, label)

	def next_batch(self, data_set, batch_size):
		data = data_set[0]
		label = data_set[1] # one-hot vectors

		data_num = np.random.choice(data.shape[0], size=batch_size, replace=False)
		batch = data[data_num,:]
		label = label[data_num,:] # one-hot vectors

		return batch, label

	def parse_layers(self, layers_str):
		layers_list_str = layers_str.split(',')

		layers_list = []
		for layer_str in layers_list_str:
			layer_dimension_list = []
			layer_dimension_list_str = layer_str.split('*')

			for layer_dimension_str in layer_dimension_list_str:
				layer_dimension_list.append(int(layer_dimension_str))

			layers_list.append(layer_dimension_list)

		return layers_list

	def calculate_num_of_weight(self, layers, pad=0, stride=1):
		layer_type = []
		num_of_weight_per_layer = []
		num_of_bias_per_layer = []
		num_of_neuron_per_layer = []

		for layer in layers:
			if layer is layers[0]:
				type = 'input' # input
				layer_type.append(type)
				num_of_neuron_per_layer.append(layer)

			elif layer is layers[-1]:
				type = 'output' # output, fully-connected
				layer_type.append(type)
				num_of_weight = np.prod(layer)*np.prod(num_of_neuron_per_layer[-1])
				num_of_weight_per_layer.append(num_of_weight)
				num_of_bias_per_layer.append(np.prod(layer))
				num_of_neuron_per_layer.append(layer)

			elif len(layer) == 4:
				type = 'conv' # convolutional
				layer_type.append(type)

				num_of_weight_per_layer.append(np.prod(layer))
				num_of_bias_per_layer.append(layer[3])

				h = (num_of_neuron_per_layer[-1][0] - layer[0] + 2*pad) / stride + 1
				w = (num_of_neuron_per_layer[-1][1] - layer[1] + 2*pad) / stride + 1
				d = layer[3]

				max_pool_f = 8
				max_pool_stride = 8

				h_max_pool = (h - max_pool_f) / max_pool_stride + 1
				#w_max_pool = (w - max_pool_f) / max_pool_stride + 1
				w_max_pool = w
				d_max_pool = d

				num_of_neuron_per_layer.append([h_max_pool,w_max_pool,d_max_pool])
				layer_type.append('max_pool')

			else:
				type = 'hidden' # fully-connected
				layer_type.append(type)
				num_of_weight = np.prod(layer)*np.prod(num_of_neuron_per_layer[-1])
				num_of_weight_per_layer.append(num_of_weight)
				num_of_bias_per_layer.append(np.prod(layer))
				num_of_neuron_per_layer.append(layer)

		#print('layer_type:', layer_type)
		#print('num_of_neuron_per_layer:', num_of_neuron_per_layer)
		#print('num_of_weight_per_layer:', num_of_weight_per_layer)
		#print('num_of_bias_per_layer:', num_of_bias_per_layer)

		return [layer_type, num_of_neuron_per_layer,
			num_of_weight_per_layer, num_of_bias_per_layer]

def main(args):
	if args.dnn_name == None or args.dnn_name == '':
		print('No dnn name. Use -dnn_name')
		return

	if not os.path.exists(args.dnn_name):
		print(args.dnn_name, 'does not exists')
		return

	fe_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
		args.dnn_name), os.path.splitext(os.path.basename(__file__))[0])
	fe_file_name = os.path.splitext(os.path.basename(__file__))[0] + '.obj' 
	fe_file_path = os.path.join(fe_dir, fe_file_name)
	#print('fe_file_path', fe_file_path)

	fe = None
	if os.path.exists(fe_file_path):
		fe = pickle.load(open(fe_file_path, 'rb'))

	if args.mode == 'c':
		print('[c] creating a function estimator')
		if args.input_size == -1:
			print('No input size. Use -input_size')
			return

		if args.output_size == -1:
			print('No output size. Use -output_size')
			return

		if args.placeholder_size == -1:
			print('No placeholder size. Use -placeholder_size')
			return

		if not args.layers:
			layers = '256,256,256'
		else:
			layers = args.layers

		func_set = FunctionSet()
		layers = str(args.input_size+args.output_size) + ',' + layers + \
			',' + str(func_set.num_of_pool)

		#print('[c] layers:', layers)
		fe = FunctionEstimator(args.dnn_name, func_set, layers,
			args.input_size, args.output_size, args.placeholder_size, fe_dir)

	if args.mode == 'g':
		print('[g] generating training data for the function estimator')
		fe.generate_training_data(args.num_of_program, args.input_data_size,
			args.input_low, args.input_high)

	elif args.mode == 't':
		print('[t] training the function estimator')

		train_data = np.load(fe.train_data_path)
		train_label = np.load(fe.train_label_path)
		test_data = np.load(fe.test_data_path)
		test_label = np.load(fe.test_label_path)
		val_data = np.load(fe.val_data_path)
		val_label = np.load(fe.val_label_path)

		train_set = [ train_data, train_label ]
		test_set = [ test_data, test_label ]
		val_set = [ val_data, val_label ]

		print('[t] data:', 'train/val.shape:',
			train_set[0].shape, val_set[0].shape)

		fe.train(train_set, val_set, args.train_batch_size, args.train_iteration)

	elif args.mode == 'e':
		print('[e] executing the function estimator')

		if args.input_data == None:
			print('[e] No input_data. Use -input_data')
			return

		if args.output_data == None:
			print('[e] No output_data. Use -output_data')
			return

		input_data = np.load(args.input_data)
		print('input_data', input_data.shape)
		output_data = np.load(args.output_data)
		print('output_data', output_data.shape)

		assert input_data.shape[0] == output_data.shape[0]
		input_data = np.reshape(input_data, [input_data.shape[0],-1])
		output_data = np.reshape(output_data, [output_data.shape[0],-1])

		input_to_estimator = np.concatenate([input_data, output_data], axis=1)
		#print('input_to_estimator', input_to_estimator.shape)
		estimator_output = fe.infer(input_to_estimator)
		#print('estimator_output.shape', estimator_output.shape)
		#print('estimator_output', estimator_output)
		distribution = np.average(estimator_output, axis=0)
		#print(distribution)
		#print('distribution.shape', distribution.shape)
		np.save(fe.distribution_file_path, distribution)
		print('function estimation completed')

		#sorted_distribution = np.sort(distribution)[::-1]
		#print('sorted_distribution', sorted_distribution)
		return

	if not os.path.exists(fe_dir):
		os.makedirs(fe_dir)
	pickle.dump(fe, open(fe_file_path, 'wb'))

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('-mode', type=str,	help='mode', default=None)
	parser.add_argument('-dnn_name', type=str, help='dnn_name', default=None)
	parser.add_argument('-layers', type=str, help='layers', default=None)
	parser.add_argument('-input_size', type=int, help='input_size', default=-1)
	parser.add_argument('-output_size', type=int, help='output_size', default=-1)
	parser.add_argument('-placeholder_size', type=int, help='placeholder_size', default=-1)
	parser.add_argument('-input_data', type=str, help='input', default=None)
	parser.add_argument('-output_data', type=str, help='output', default=None)
	parser.add_argument('-num_of_program', type=int, help='num_of_program', default=100)
	parser.add_argument('-input_data_size', type=int, help='input_data_size', default=1000)
	parser.add_argument('-train_iteration', type=int, help='train_iteration', default=10000)
	parser.add_argument('-train_batch_size', type=int, help='train_batch_size', default=100)
	parser.add_argument('-input_low', type=float, help='input_low', default=0.0)
	parser.add_argument('-input_high', type=float, help='input_high', default=1.0)

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
