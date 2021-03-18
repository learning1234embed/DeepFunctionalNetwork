from __future__ import print_function
import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import copy
import os
import random
import pickle

from function import Function, FunctionSet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#gpu_options = tf.GPUOptions(allow_growth = True)
gpu_options = None

class Individual:
	def __init__(self, genotype):
		self.genotype = genotype
		self.DiGraph = None
		self.valid_function = None
		self.connection_weight = None
		self.connection_bias = None
		self.output_contribution_sum_list = None
		self.input_contribution_sum_list = None
		self.active_function_sequence_instance = None

		self.update_graph()
		self.update_valid_function()
		self.init_connection_weight(random_init=False)

	def init_connection_weight(self, random_init=False):
		input_table = self.genotype[1]
		connection_weight = []
		connection_bias = []
		for input_list in input_table:
			if random_init == True:
				# Xavier intilization
				ni = len(input_list)
				no = len(input_list)
				w = np.random.random(len(input_list)).astype('f')*np.sqrt(1/float(ni+no))
				b = np.random.random(len(input_list)).astype('f')*np.sqrt(1/float(ni+no))
			else:
				w = np.ones(len(input_list), dtype=np.float32)
				b = np.repeat(0.0000001, len(input_list)).astype('f')
			connection_weight.append(w)
			connection_bias.append(w)

		self.connection_weight = connection_weight
		self.connection_bias = connection_bias

	def update_graph(self):
		input_node_name = "input"
		output_node_name = "output"
		self.DiGraph = nx.DiGraph()
		self.DiGraph.clear()
		self.DiGraph.add_node(input_node_name)

		active_function_sequence = self.genotype[0]
		active_function_sequence_set = list(set(copy.deepcopy(active_function_sequence)))
		active_function_sequence_instance = [None] * len(active_function_sequence)

		for i in range(len(active_function_sequence_set)):
			idx = 0
			for j in range(len(active_function_sequence)):
				if active_function_sequence_set[i] == active_function_sequence[j]:
					active_function_sequence_instance[j] = active_function_sequence_set[i] + str('-') + str(idx)
					idx += 1
		self.active_function_sequence_instance = active_function_sequence_instance

		for i in range(len(active_function_sequence_instance)):
			self.DiGraph.add_node(active_function_sequence_instance[i])

		input_function_table = self.genotype[3]
		for j in range(len(input_function_table)):
			parent_nodes = input_function_table[j]
			for i in parent_nodes:
				if j == len(input_function_table)-1:
					if i == -1:
						parent = input_node_name
					else:
						parent = active_function_sequence_instance[i]
					child = output_node_name
				elif i == -1:
					parent = input_node_name
					child = active_function_sequence_instance[j]
					self.DiGraph.add_edge(input_node_name, active_function_sequence_instance[j])
				else:
					parent = active_function_sequence_instance[i]
					child = active_function_sequence_instance[j]

				self.DiGraph.add_edge(parent, child)

		# remove leaves
		leaves = [x for x in self.DiGraph.nodes() if self.DiGraph.out_degree(x) == 0]
		for leaf in leaves:
			if leaf != output_node_name:
				self.DiGraph.remove_node(leaf)
		while len(leaves) > 1:
			leaves = [x for x in self.DiGraph.nodes() if self.DiGraph.out_degree(x) == 0]
			for leaf in leaves:
				if leaf != output_node_name:
					self.DiGraph.remove_node(leaf)

	def draw_plain_graph(self, filename='plain_graph.png'):
		self.update_graph()
		DiGraph = copy.deepcopy(self.DiGraph)

		node_name_short = {}
		for node in DiGraph.nodes():
			node_name_short[node] = node.split('-')[0]
			#node_name_short[node] = node.split(':')[0]
			#node_name_short[node] = node
		DiGraphShort = nx.relabel_nodes(DiGraph, node_name_short, copy=True)
		print(DiGraphShort.nodes())

		pos = graphviz_layout(DiGraphShort, prog='dot')
		nx.draw(DiGraphShort, pos, with_labels=True,
			node_color='#ADBDDA', width=1.5, edge_color='#4A4A4A',
			node_shape='o',	node_size=2000, font_size=20)

		for node in DiGraphShort.nodes():
			if node == 'input' or node == 'output':
				nx.draw_networkx_nodes(self.DiGraph, nodelist=[node], pos=pos,
					node_color='#C0C0C0',
					node_shape='o',	node_size=2000, font_size=20)

			elif "neuron" in node:
				nx.draw_networkx_nodes(self.DiGraph, nodelist=[node], pos=pos,
					node_color='#B99DCA',
					node_shape='o',	node_size=2000, font_size=20)

		plt.savefig(filename)
		plt.clf()

	def draw_interpretable_graph(self, filename='interpretable_graph.png'):
		active_function_sequence = self.genotype[0]
		input_table = self.genotype[1]
		output_table = self.genotype[2]
		input_function_table = self.genotype[3]
		input_len_table = self.genotype[4]
		active_function_sequence_instance = self.active_function_sequence_instance

		output_sum = np.sum(self.output_contribution_sum_list)
		output_ratio_list = np.asarray(self.output_contribution_sum_list) / output_sum

		input_sum = np.sum(np.sum(self.input_contribution_sum_list))
		input_ratio_list = []
		for s in self.input_contribution_sum_list:
			input_ratio_list.append(list(s/input_sum))

		pos = graphviz_layout(self.DiGraph, prog='dot')
		nx.draw_networkx_labels(self.DiGraph, pos=pos)

		nodes = list(self.DiGraph.nodes)
		for node in nodes:
			node_size = 600
			node_shape = 'o'
			linewidths = 50
			node_color = 'A2FF'
			if node != 'input' and node != 'output':
				function_num = None
				for i in range(len(active_function_sequence_instance)):
					if node == active_function_sequence_instance[i]:
						function_num = i
						break
				assert function_num != None
				#node_size *= output_ratio_list[function_num]
				linewidths *= output_ratio_list[function_num]
				node_color = hex(int(255*output_ratio_list[function_num]))[2:] + '0000'
				if len(node_color) == 5:
					node_color = '#0' + node_color
				else:
					node_color = '#' + node_color

			else:
				node_shape = 's'
				node_size = 1600
				linewidths = 1
				node_color = '#000000'

			nx.draw_networkx_nodes(self.DiGraph, nodelist=[node], pos=pos,
				node_size=node_size, node_shape=node_shape,
				node_color='#DCDCDC', linewidths=linewidths)
				#node_color=node_color, linewidths=linewidths)

		active_function_sequence_instance.append('output')
		active_function_sequence_instance.append('input')

		edges = list(self.DiGraph.edges)
		for edge in edges:
			edge_list = list(edge)
			width = 10.0
			min_width = 1.0

			#input_ratio_array = np.asarray(input_ratio_list)
			input_ratio_array = []
			for l in input_ratio_list:
				    input_ratio_array += l

			input_ratio_array = np.asarray(input_ratio_array)
			nonzero_input_ratio_array = input_ratio_array[np.nonzero(input_ratio_array)]

			min_ratio = np.amin(nonzero_input_ratio_array)
			factor = min_width / min_ratio

			source = edge_list[0]
			source_func_num = None
			for i in range(len(active_function_sequence_instance)):
				if active_function_sequence_instance[i] == source:
					source_func_num = i
					break
			assert source_func_num != None
			if source_func_num == len(active_function_sequence_instance)-1:
				source_func_num = -1

			dest = edge_list[1]
			dest_func_num = None
			for i in range(len(active_function_sequence_instance)):
				if active_function_sequence_instance[i] == dest:
					dest_func_num = i
					break
			assert dest_func_num != None

			source_idx = None
			for i in range(len(input_function_table[dest_func_num])):
				if source_func_num == input_function_table[dest_func_num][i]:
					source_idx = i
					break
			assert source_idx != None

			width = 1.0
			nx.draw_networkx_edges(self.DiGraph, edgelist=[edge], pos=pos,
				width=width)

		plt.axis('off')
		plt.savefig(filename)
		plt.clf()

	def update_valid_function(self):
		if self.DiGraph is None:
			self.update_graph()

		all_nodes = self.DiGraph.nodes()
		active_function_sequence = self.genotype[0]
		valid_function = np.zeros(len(active_function_sequence), dtype=np.int32)

		for node in all_nodes:
			if node == 'input' or node == 'output':
				continue

			split = node.split('-')
			idx = 0
			for i in range(len(active_function_sequence)):
				if split[0] == active_function_sequence[i]:
					if idx == int(split[1]):
						valid_function[i] = 1
						break
					idx += 1

		self.valid_function = valid_function

class DFN:
	def __init__(self, name, input_size, output_size, row_size, column_size,
		primitive_function=None, primitive_function_distribution=None, use_neuron_function=1,
		tf_batch_size=2000, hall_of_fame_size=4, fitness='mse', loss='cross_entropy'):
		self.name = name
		if primitive_function is None:
			self.primitive_function = []
		else:
			self.primitive_function = primitive_function
		if primitive_function_distribution is None:
			self.primitive_function_distribution = \
				[float(1)/float(len(self.primitive_function))] * len(self.primitive_function)
		else:
			self.primitive_function_distribution = primitive_function_distribution
		self.use_neuron_function = use_neuron_function
		if use_neuron_function == 1 and primitive_function_distribution is not None:
			avg_distribution = np.mean(self.primitive_function_distribution)
			dist = np.append(self.primitive_function_distribution, avg_distribution)
			self.primitive_function_distribution = dist / np.sum(dist)
		self.input_size = input_size
		self.output_size = output_size
		self.row_size = row_size
		self.column_size = column_size
		self.population = []
		self.hall_of_fame = None
		self.hall_of_fame_fitness_score = None
		self.hall_of_fame_size = hall_of_fame_size
		self.input_base_name = 'input'
		self.weighted_input_base_name = 'weighted_input'
		self.output_base_name = 'output'
		self.connection_weight_base_name = 'connection_weight'
		self.connection_bias_base_name = 'connection_bias'
		self.function_weight_base_name = 'function_weight'
		self.tf_batch_size = tf_batch_size
		self.fitness = fitness
		self.loss = loss

	def mse(self, output1, output2):
		assert output1.shape == output2.shape
		# MSE: mean squared error
		mse = np.average(np.square(output1 - output2))
		return mse

	def cross_entropy(self, output1, output2):
		cross_entropy = np.mean(-np.sum(np.multiply(output2, np.log(output1+0.00000000001)),
			axis=1))
		return cross_entropy

	def fitness_function(self, output1, output2):
		if self.fitness == 'mse':
			return self.mse(output1, output2)
		elif self.fitness == 'cross_entropy':
			return self.cross_entropy(output1, output2)

	def mse_tf(self, output, output_tensor_, name):
		mse = tf.reduce_mean(tf.square(output - output_tensor_), name=name)
		return mse

	def cross_entropy_tf(self, output, output_tensor_, name):
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_tensor_ * tf.log(output),
			reduction_indices=[1]), name=name)
		return cross_entropy

	def fitness_function_tf(self, output, output_tensor_, name):
		if self.fitness == 'mse':
			return self.mse_tf(output, output_tensor_, name)
		elif self.fitness == 'cross_entropy':
			return self.cross_entropy_tf(output, output_tensor_, name)

	def loss_function_tf(self, output, output_tensor_, name):
		if self.loss == 'mse':
			return mse_tf(output, output_tensor_, name)
		elif self.loss == 'cross_entropy':
			return self.cross_entropy_tf(output, output_tensor_, name)

	def generate_population(self, population_num):
		for i in range(population_num):
			self.population.append(self.generate_individual())

	def next_batch(self, input_data, output_data, batch_size):
		assert input_data.shape[0] == output_data.shape[0]
		data_num = np.random.choice(input_data.shape[0], size=batch_size,
			replace=False)
		batch_input_data = input_data[data_num,:]
		batch_output_data = output_data[data_num,:]
		return batch_input_data, batch_output_data

	def save_connection_weight(self, individual, graph, sess):
		active_function_sequence = individual.genotype[0]
		active_function = individual.genotype[5]
		connection_weight_list = []
		connection_bias_list = []
		function_weight_list = []

		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				connection_weight = graph.get_tensor_by_name(self.connection_weight_base_name+str(i)+':0')
				connection_weight_list.append(connection_weight)
				connection_bias = graph.get_tensor_by_name(self.connection_bias_base_name+str(i)+':0')
				connection_bias_list.append(connection_bias)
				function_name = active_function_sequence[i]
				func = active_function[i]
				if func.parameter != None and func.parameter.get('weight') is not None:
					function_weight = graph.get_tensor_by_name(self.function_weight_base_name+str(i)+':0')
					function_weight_list.append(function_weight)

		output_weight = graph.get_tensor_by_name(self.connection_weight_base_name+str(len(individual.valid_function))+':0')
		connection_weight_list.append(output_weight)
		output_bias = graph.get_tensor_by_name(self.connection_bias_base_name+str(len(individual.valid_function))+':0')
		connection_bias_list.append(output_bias)

		updated_connection_weight = sess.run(connection_weight_list)
		updated_connection_bias = sess.run(connection_bias_list)
		updated_function_weight = sess.run(function_weight_list)

		idx = 0
		function_idx = 0
		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				individual.connection_weight[i] = updated_connection_weight[idx]
				individual.connection_bias[i] = updated_connection_bias[idx]
				function_name = active_function_sequence[i]
				func = active_function[i]
				idx += 1

				if func.parameter != None and func.parameter.get('weight') is not None:
					func.parameter['weight'] = updated_function_weight[function_idx]
					function_idx += 1

		individual.connection_weight[-1] = updated_connection_weight[-1]
		individual.connection_bias[-1] = updated_connection_bias[-1]

	def function_contribution_tf(self, individual, input_data, output_data, target='fitness'):
		active_function_sequence = individual.genotype[0]
		input_table = individual.genotype[1]
		output_table = individual.genotype[2]
		input_function_table = individual.genotype[3]
		input_len_table = individual.genotype[4]

		tf.reset_default_graph()
		self.build_individual_network_tf(individual)
		graph = tf.get_default_graph()

		if target == 'fitness':
			target_t = graph.get_tensor_by_name('fitness:0')
		elif target == 'loss':
			target_t = graph.get_tensor_by_name('loss:0')

		output_contribution_sum_list = []
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())

			# funciton output
			for i in range(len(individual.valid_function)):
				if individual.valid_function[i] == 1:
					output_t = graph.get_tensor_by_name(self.output_base_name+str(i)+':0')
					contribution = self.compute_contribution_tf(graph, sess,
						individual, input_data, output_data,
						target_t, output_t)
					output_sesitivity_sum = np.sum(contribution)
					output_contribution_sum_list.append(output_sesitivity_sum)

		tf.reset_default_graph()

		individual.output_contribution_sum_list = output_contribution_sum_list
		return output_contribution_sum_list

	def connection_contribution_tf(self, individual, input_data, output_data, target='fitness'):
		active_function_sequence = individual.genotype[0]
		input_table = individual.genotype[1]
		output_table = individual.genotype[2]
		input_function_table = individual.genotype[3]
		input_len_table = individual.genotype[4]

		tf.reset_default_graph()
		self.build_individual_network_tf(individual)
		graph = tf.get_default_graph()

		if target == 'fitness':
			target_t = graph.get_tensor_by_name('fitness:0')
		elif target == 'loss':
			target_t = graph.get_tensor_by_name('loss:0')

		input_contribution_list = []
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())

			# connection
			for i in range(len(input_table)-1):
				if individual.valid_function[i] == 1:
					output_t = graph.get_tensor_by_name(self.weighted_input_base_name+str(i)+':0')
					contribution = self.compute_contribution_tf(graph, sess,
						individual, input_data, output_data,
						target_t, output_t)
					input_contribution = np.sum(contribution,axis=0)
					input_contribution_list.append(input_contribution)
				else:
					input_contribution_list.append(np.zeros(len(input_table[i])))

			output_t = graph.get_tensor_by_name('output:0')
			contribution = self.compute_contribution_tf(graph, sess,
				individual, input_data, output_data, target_t, output_t)
			input_contribution = np.sum(contribution,axis=0)
			input_contribution_list.append(input_contribution)

		input_contribution_sum_list = []
		for i in range(len(input_len_table)):
			input_contribution_sum = []
			idx = 0
			for j in range(len(input_len_table[i])):
				sum = np.sum(input_contribution_list[i][idx:idx+input_len_table[i][j]])
				input_contribution_sum.append(sum)
				idx += input_len_table[i][j]
			input_contribution_sum_list.append(input_contribution_sum)

		output_input_contribution_sum = [0] * len(input_function_table[-1])
		for i in range(len(input_table[-1])):
			if input_table[-1][i] < self.input_size:
				output_input_contribution_sum[0] += input_contribution_list[-1][i]
			else:
				idx = None
				for j in range(len(input_function_table[-1])):
					if input_function_table[-1][j] == -1:
						continue
					if input_table[-1][i] in output_table[input_function_table[-1][j]]:
						idx = j
						break
				assert idx != None
				#if -1 in input_function_table[-1]:
				#	idx += 1
				output_input_contribution_sum[idx] += input_contribution_list[-1][i]

		input_contribution_sum_list.append(output_input_contribution_sum)

		tf.reset_default_graph()

		individual.input_contribution_sum_list = input_contribution_sum_list
		return input_contribution_sum_list

	def compute_contribution_tf(self, graph, sess, individual, input_data, output_data,
		target_tensor, output_tensor):
		assert input_data.shape[0] == output_data.shape[0]

		input_tensor = graph.get_tensor_by_name(self.output_base_name + '-1:0')
		output_tensor_ = graph.get_tensor_by_name('output_tensor_:0')

		contribution_set = []
		for i in range(input_data.shape[0]):
			input_vector = input_data[i:i+1]
			output_vector = output_data[i:i+1]
			gradient = tf.gradients(target_tensor, output_tensor,
				unconnected_gradients='zero')
			second_order_approximate = tf.square(gradient[0])
			contribution = second_order_approximate * tf.square(output_tensor)
			contribution_information = sess.run(contribution,
				feed_dict={input_tensor:input_vector,
				output_tensor_:output_vector})
			contribution_set.append(contribution_information[0])

		contribution_set = np.stack(contribution_set)
		return contribution_set

	def compute_jacobian_tf(self, individual, input_data, output_idx=None):
		tf.reset_default_graph()
		self.build_individual_network_tf(individual)
		graph = tf.get_default_graph()
		input_tensor = graph.get_tensor_by_name(self.output_base_name + '-1:0')
		output_tensor = graph.get_tensor_by_name('output:0')

		if input_data.ndim == 1:
			input_data = np.expand_dims(input_data, axis=0)

		jacobian_set = []
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(input_data.shape[0]):
				input_vector = input_data[i:i+1]
				gradient_list = []

				if output_idx == None:
					output_idx = range(self.output_size)

				for idx in output_idx:
					gradient = tf.gradients(output_tensor[0,idx],
						input_tensor,
						unconnected_gradients='zero')
					gradient_list.append(gradient[0][0])

					jacobian = sess.run(gradient_list,
						feed_dict={input_tensor:input_vector})

				jacobian_set.append(np.stack(jacobian))

		tf.reset_default_graph()
		jacobian_set = np.stack(jacobian_set)
		return jacobian_set

	def update_weight_tf(self, individual, train_data, train_label, val_data, val_label,
		iteration=10000, batch_size=100, save=False):

		tf.reset_default_graph()
		regularization = self.build_individual_network_tf(individual)
		graph = tf.get_default_graph()
		input_tensor = graph.get_tensor_by_name(self.output_base_name + '-1:0')
		output_tensor = graph.get_tensor_by_name('output:0')
		output_tensor_ = graph.get_tensor_by_name('output_tensor_:0')
		fitness = graph.get_tensor_by_name('fitness:0')
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
			name='optimizer').minimize(fitness+regularization)

		input_tensor_shape = input_tensor.get_shape().as_list()
		train_data_reshaped = np.reshape(train_data,
				[-1, np.prod(input_tensor_shape[1:])])

		val_data_reshaped = np.reshape(val_data,
			[-1, np.prod(input_tensor_shape[1:])])
		val_iteration = val_data.shape[0] // self.tf_batch_size
		val_remained = val_data.shape[0] % self.tf_batch_size

		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())
			lowest_fitness = 0
			for i in range(iteration):
				input_batch, output_batch = self.next_batch(train_data_reshaped,
					train_label, batch_size)

				if i % (100) == 0 or i == (iteration-1):
					f_sum = 0
					for j in range(val_iteration):
						f = sess.run(fitness, feed_dict={
							input_tensor:val_data_reshaped[j*self.tf_batch_size:j*self.tf_batch_size+self.tf_batch_size],
							output_tensor_: val_label[j*self.tf_batch_size:j*self.tf_batch_size+self.tf_batch_size]})
						f_sum += f*self.tf_batch_size

					if val_remained > 0:
						f = sess.run(fitness, feed_dict={
							input_tensor:val_data_reshaped[val_iteration*self.tf_batch_size:val_iteration*self.tf_batch_size+val_remained],
							output_tensor_: val_label[val_iteration*self.tf_batch_size:val_iteration*self.tf_batch_size+val_remained]})
						f_sum += f*val_remained

					f_avg = f_sum / val_data.shape[0]
					print("step %d, validation fitness: %f" % (i, f_avg))

					if i == 0:
						lowest_fitness = f_avg
					else:
						if f_avg < lowest_fitness:
							lowest_fitness = f_avg
							if save:
								print('save weight for', f_avg)
								self.save_connection_weight(individual, graph, sess)

				sess.run(optimizer, feed_dict={input_tensor: input_batch,
					output_tensor_: output_batch})

		tf.reset_default_graph()

	def build_individual_network_tf(self, individual):
		if individual.valid_function is None:
			individual.update_valid_function()

		active_function_sequence = individual.genotype[0]
		input_table = individual.genotype[1]
		output_table = individual.genotype[2]
		input_function_table = individual.genotype[3]
		input_len_table = individual.genotype[4]
		active_function = individual.genotype[5]
		function_instance_num = individual.genotype[6]

		tf.reset_default_graph()
		graph = tf.get_default_graph()

		input_tensor = tf.placeholder(tf.float32, [None, self.input_size],
			name=self.output_base_name + '-1')

		weight_to_regularize = []

		vector_list = [ input_tensor ]
		output_idx = copy.deepcopy(input_table[-1])

		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				function_name = active_function_sequence[i]
				#func = self.function_dict[function_name]
				func = active_function[i]
				input_list = []
				input_pos = 0
				for j in range(len(input_function_table[i])):
					input_function_num = input_function_table[i][j]
					target_input_tensor_name = self.output_base_name + str(input_function_num) + ":0"
					target_input = graph.get_tensor_by_name(target_input_tensor_name)
					start_idx = input_table[i][input_pos]
					if input_function_num != -1:
						offset = output_table[input_function_num][0]
						start_idx -= offset

					end_idx = start_idx + input_len_table[i][j]
					extracted_input = target_input[:,start_idx:end_idx]
					input_list.append(extracted_input)
					input_pos += input_len_table[i][j]

				connection_weight = tf.Variable(individual.connection_weight[i],
					name=self.connection_weight_base_name + str(i))

				weight_to_regularize.append(connection_weight)
				connection_bias = tf.Variable(individual.connection_bias[i],
					name=self.connection_bias_base_name + str(i))
				weight_to_regularize.append(connection_bias)
				input_t = tf.concat(input_list, 1,
					name=self.input_base_name + str(i))
				weighted_input_t = tf.add(tf.multiply(input_t,
					connection_weight), connection_bias,
					name=self.weighted_input_base_name + str(i))
				if func.parameter != None and func.parameter.get('weight') is not None:
					function_weight = tf.Variable(func.parameter.get('weight'),
						name=self.function_weight_base_name + str(i))
					weight_to_regularize.append(function_weight)
				else:
					function_weight = None
				output_original = func.tf_exec(weighted_input_t,
					weight=function_weight)
				output_t = tf.identity(output_original,
					name=self.output_base_name + str(i))
				vector_list.append(output_t)
			else:
				offset = len(output_table[i])
				for k in range(len(input_table[-1])):
					if input_table[-1][k] > output_table[i][-1]:
						output_idx[k] -= offset

		vector = tf.concat(vector_list, 1)
		connection_weight = tf.Variable(individual.connection_weight[-1],
			name=self.connection_weight_base_name + str(len(active_function_sequence)))
		weight_to_regularize.append(connection_weight)
		connection_bias = tf.Variable(individual.connection_bias[-1],
			name=self.connection_bias_base_name + str(len(active_function_sequence)))
		weight_to_regularize.append(connection_bias)
		gathered_output = tf.gather(vector, output_idx, axis=1,
			name='gathered_output')
		#output = tf.add(tf.multiply(gathered_output, connection_weight),
		#	connection_bias, name='output')
		output = tf.nn.softmax(tf.add(tf.multiply(gathered_output, connection_weight),
			connection_bias), name='output')

		output_tensor_ = tf.placeholder(tf.float32, [None, self.output_size],
			name='output_tensor_')

		fitness = self.fitness_function_tf(output, output_tensor_, 'fitness')
		loss = self.loss_function_tf(output, output_tensor_, 'loss')

		#regularization = 0.000001 * tf.nn.l2_loss(tf.concat(weight_to_regularize, 0))
		weight_concat = tf.concat([tf.reshape(weight,
			[-1]) for weight in weight_to_regularize], 0)
		regularization = 0.000001 * tf.nn.l2_loss(weight_concat)
		return regularization

	def execute_individual_tf(self, individual, input_data):
		self.build_individual_network_tf(individual)
		graph = tf.get_default_graph()
		input_tensor = graph.get_tensor_by_name(self.output_base_name + '-1:0')
		output_tensor = graph.get_tensor_by_name('output:0')

		if input_data.ndim == 1:
			input_data = np.expand_dims(input_data, axis=0)

		input_data = np.reshape(input_data, (input_data.shape[0], self.input_size))

		iteration = input_data.shape[0] // self.tf_batch_size
		remained = input_data.shape[0] % self.tf_batch_size
		output_value_list = []

		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(iteration):
				output_value = sess.run(output_tensor,
					feed_dict={input_tensor:input_data[i*self.tf_batch_size:i*self.tf_batch_size+self.tf_batch_size]})
				output_value_list.append(output_value)

			if remained > 0:
				output_value = sess.run(output_tensor,
					feed_dict={input_tensor:input_data[iteration*self.tf_batch_size:iteration*self.tf_batch_size+remained]})
				output_value_list.append(output_value)

		tf.reset_default_graph()
		output_value_set = np.vstack(output_value_list)
		return output_value_set

	def freeze_individual_tf(self, individual, file_path):
		self.build_individual_network_tf(individual)
		graph = tf.get_default_graph()

		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())
			frozen_graph = tf.graph_util.convert_variables_to_constants(
				sess, graph.as_graph_def(), ['output'])
			with open(file_path, 'wb') as f:
				f.write(frozen_graph.SerializeToString())

		tf.reset_default_graph()

	def execute_individual_python(self, individual, input_vector):
		if individual.valid_function is None:
			individual.update_valid_function()

		input_table = individual.genotype[1]
		output_table = individual.genotype[2]
		active_function = individual.genotype[5]
		vector = np.empty(output_table[-1][-1] + 1)
		vector[:] = np.nan

		# input vector
		vector_pos = 0
		vector[vector_pos:vector_pos+len(input_vector)] = input_vector

		active_function_sequence = individual.genotype[0]

		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				function_name = active_function_sequence[i]
				func = active_function[i]
				input_data = vector[input_table[i]]
				if len(input_data) != func.input_size: 
					raise Exception('len(input_data) != func.input_size') 
				if np.any(np.isnan(input_data)):
					raise Exception('NaN input data') 
				connection_weight = individual.connection_weight[i]
				connection_bias = individual.connection_bias[i]
				output_data = func.py_exec(np.add(np.multiply(input_data,
					connection_weight), connection_bias))
				if len(output_data) != func.output_size: 
					raise Exception('len(output_data) != func.output_size') 
				if len(output_data) != len(output_table[i]): 
					raise Exception('len(output_data) != func.output_size') 
				vector[output_table[i]] = output_data 

		output = np.add(np.multiply(vector[input_table[-1]],
			individual.connection_weight[-1]), individual.connection_bias[-1])
		if np.any(np.isnan(output)):
			raise Exception('NaN output')

		return output

	def sort_population(self, population, fitness_score_list):
		fitness_score = [item[0] for item in fitness_score_list]
		sorted_idx = np.argsort(fitness_score)
		sorted_fitness_score_list = [fitness_score_list[idx] for idx in sorted_idx]
		sorted_population = [population[idx] for idx in sorted_idx]
		return sorted_population, sorted_fitness_score_list

	def run(self, generation, num_of_mutation_per_individual, input_data, output_data,
		jacobian_idx=None, jacobian_data=None):

		input_data = np.reshape(input_data, (input_data.shape[0], self.input_size))

		for i in range(generation):
			population_score = []
			for j in range(len(self.population)):
				individual_score = []
				individual_copy = [None] * num_of_mutation_per_individual
				for k in range(num_of_mutation_per_individual):
					individual_copy[k] = copy.deepcopy(self.population[j])
					self.mutate_individual(individual_copy[k])
					fitness_score = self.evaluate_individual(individual_copy[k],
						input_data, output_data, jacobian_idx, jacobian_data)
					individual_score.append(fitness_score)

				individual_copy.append(copy.deepcopy(self.population[j]))
				fitness_score = self.evaluate_individual(self.population[j],
					input_data, output_data, jacobian_idx, jacobian_data)
				individual_score.append(fitness_score)
				individual_copy, sorted_score = self.sort_population(individual_copy,
					individual_score)

				print('[%d][%d] best_fitnes:' % (i,j), sorted_score[0])
				self.population[j] = individual_copy[0]
				population_score.append(sorted_score[0])

			self.population, sorted_score = self.sort_population(self.population,
				population_score)
			self.hall_of_fame = self.population[0:self.hall_of_fame_size]
			self.hall_of_fame_fitness_score = sorted_score[0:self.hall_of_fame_size]

			print('[%d] hall_of_fame[0] fitness:' % (i),
				self.hall_of_fame_fitness_score[0])

			valid_function = self.hall_of_fame[0].valid_function
			valid_func_idx = np.where(valid_function == 1)[0]
			valid_func_name = [self.hall_of_fame[0].genotype[0][idx] for idx in valid_func_idx]
			print('[%d] valid function:' % (i), valid_func_name)

			#if i % 10 == 0:
			#print('Save', self.name)
			#pickle.dump(self, open(self.name, 'wb'))

	def evaluate_individual(self, individual, input_data, output_data,
		jacobian_idx=None, jacobian_data=None):
		fitness_score = 0

		output = self.execute_individual_tf(individual, input_data)
		output_score = self.fitness_function(output, output_data)

		fitness_score += output_score

		jacobian_score = -1
		if jacobian_idx is not None and jacobian_idx is not None:
			jacobian = self.compute_jacobian_tf(individual,
				input_data[jacobian_idx,:])
			jacobian_score = self.fitness_function(jacobian, jacobian_data)
			fitness_score += jacobian_score

		return [fitness_score, output_score, jacobian_score]

	def mutate_individual(self, individual):
		is_active = False
		max_num_of_try = 10
		num_of_try = 0
		while is_active == False:
			mutated_genotype, is_active = self.mutate_genotype(individual.valid_function, individual.genotype)

			if mutated_genotype and is_active == True:
				individual.genotype = mutated_genotype
				individual.update_graph()
				individual.update_valid_function()
				individual.init_connection_weight(random_init=False)
				break

			num_of_try += 1
			if num_of_try >= max_num_of_try:
				#individual.update_graph()
				#individual.update_valid_function()
				individual.init_connection_weight(random_init=False)
				break

	def generate_individual(self):
		genotype = self.generate_genotype()
		while genotype is None:
			genotype = self.generate_genotype()
		is_valid = self.check_genotype(genotype)
		while is_valid is False:
			genotype = self.generate_genotype()
			is_valid = self.check_genotype(genotype)
		return Individual(genotype)

	def check_genotype(self, genotype):
		# genotype sanity check
		active_function_sequence = genotype[0]
		input_table = genotype[1]
		output_table = genotype[2]
		input_function_table = genotype[3]
		input_len_table = genotype[4]
		active_function = genotype[5]
		function_instance_num = genotype[6]

		previous = None
		current = 0
		for i in range(len(output_table)):
			for j in range(len(output_table[i])):
				current = output_table[i][j]
				if previous == None:
					if current != self.input_size:
						print('current != self.input_size')
						return False
					previous = self.input_size
					continue

				if current != previous + 1:
					print('current != previous + 1')
					return False

				previous = current

		for i in range(len(input_table)-1):
			input_len = 0
			idx = 0
			input_function_num = input_function_table[i][idx]
			for j in range(len(input_table[i])):
				input_len += 1
				if input_len > input_len_table[i][idx]:
					input_len = 1
					idx += 1
					input_function_num = input_function_table[i][idx]

				if input_function_num == -1:
					min_len = 0
					max_len = self.input_size - 1
				else:
					min_len = output_table[input_function_num][0]
					max_len = output_table[input_function_num][-1]

				if input_table[i][j] < min_len:
					print('input_function_num', input_function_num)
					print('input_table[%d][%d] < %d' % (i, j, min_len))
					return False
				if input_table[i][j] > max_len:
					print('input_function_num', input_function_num)
					print('input_table[%d][%d] > %d' % (i, j, max_len))
					return False

		for i in range(len(active_function_sequence)):
			#function = self.function_dict[active_function_sequence[i]]
			function = active_function[i]

			if len(output_table[i]) != function.output_size:
				print('len(output_table[%d] != function.output_size)' % i)
				print('%d != %d' % (len(output_table[i]), function.output_size))
				return False

			if len(input_table[i]) != function.input_size:
				print('len(input_table[%d] != function.intput_size)' % i)
				return False

			if any(input_table[i]) >= min(output_table[i]):
				print('any(input_table[%d]) < min(output_table[i])' % i)
				return False

		for i in range(len(input_len_table)):
			if len(input_len_table[i]) != len(input_function_table[i]):
				print('input_len_table[i]) != len(input_function_table[i]')
				return False

			if np.sum(input_len_table[i]) != len(input_table[i]):
				print('np.sum(input_len_table[i]) != len(input_table[i])')
				return False

		if len(input_function_table[-1]) != len(input_function_table[-1]):
			print('len(input_function_table[-1]) != len(input_function_table[-1])')
			return False

		output_input_set = []
		for i in input_table[-1]:
			input_function_num = 0
			if i < self.input_size:
				input_function_num = -1
			else:
				for j in range(len(output_table)):
					if i <= output_table[j][-1]:
						input_function_num = j
						break
			output_input_set.append(input_function_num)

		output_input_set = set(output_input_set)

		if len(set(input_function_table[-1]).difference(output_input_set)) != 0:
			print('len(set(input_function_table[-1].difference(output_input_set)) != 0')
			return False

		return True
	
	def mutate_genotype(self, valid_function, genotype):
		active_function_sequence = genotype[0]
		input_table = genotype[1]
		output_table = genotype[2]
		input_function_table = genotype[3]
		input_len_table = genotype[4]

		gene = []
		for i in range(len(active_function_sequence)):
			gene += [active_function_sequence[i]]
		#	gene += input_function_table[i]
		#gene += input_table[-1]

		for i in range(len(input_function_table)):
			gene += input_function_table[i]

		to_mutate = np.random.randint(len(gene))

		is_active = False

		if isinstance(gene[to_mutate], str):
			mutate_type = 'function'
			if valid_function[to_mutate] == 1:
				is_active = True
		elif to_mutate >= len(gene) - len(input_function_table[-1]):
			mutate_type = 'output'
			is_active = True
		else:
			mutate_type = 'input_function'

			input_function_num, input_function_idx = \
				self.find_function_num_by_talbe_idx(genotype, to_mutate-len(active_function_sequence)+1)
			if valid_function[input_function_num] == 1:
				is_active = True

		genotype_clone = copy.deepcopy(genotype)
		mutated_genotype = None

		if mutate_type == 'function':
			new_primitive_function_num = np.random.choice(len(self.primitive_function), 1,
				p=self.primitive_function_distribution)[0]
			mutated_genotype = self.replace_function(genotype_clone,
				to_mutate, new_primitive_function_num)
			#if mutated_genotype:
			#	print('replace_function success')

		elif mutate_type == 'output':
			mutated_genotype = self.replace_output(genotype_clone, gene, to_mutate)
		elif mutate_type == 'input_function':
			# connection (input_function) mutate
			input_function_mutate_type = np.random.randint(3)
			#input_function_mutate_type = 2
			num_of_try = 1
			if input_function_mutate_type == 0:
				#  mutate 1. replace input connection
				for _ in range(num_of_try):
					mutated_genotype = self.replace_input_function(genotype_clone, gene, to_mutate)
					if mutated_genotype:
						break
			elif input_function_mutate_type == 1:
				#  mutate 2. add input connection
				for _ in range(num_of_try):
					mutated_genotype = self.add_input_function(genotype_clone, gene, to_mutate)
					if mutated_genotype:
						break
			elif input_function_mutate_type == 2:
				#  mutate 3. remove input connection
				for _ in range(num_of_try):
					mutated_genotype = self.remove_input_function(genotype_clone, gene, to_mutate)
					if mutated_genotype:
						break

			if mutated_genotype is None:
				return None, is_active

		if mutated_genotype is None:
			return None, is_active

		if self.check_genotype(mutated_genotype) is False:
			raise Exception('SANITY CHECK FAILED!')

		return mutated_genotype, is_active

	def find_function_num_by_talbe_idx(self, genotype, idx):
		active_function_sequence = genotype[0]
		input_function_table = genotype[3]

		input_function_num = None
		input_function_idx = None
		input_sum = 0

		for i in range(len(input_function_table)):
			for j in range(len(input_function_table[i])):
				input_sum += 1
				if input_sum == idx:
					input_function_num = i
					input_function_idx = j
					break

			if input_function_num and input_function_idx:
				break

		return input_function_num, input_function_idx

	def find_function_num_by_input(self, genotype, input_value):
		output_table = genotype[2]
		input_function_num = None

		if input_value < self.input_size:
			input_function_num = -1
		else:
			for i in range(len(output_table)):
				if input_value in output_table[i]:
					input_function_num = i
					break

		return input_function_num

	def replace_function(self, genotype, old_function_num, new_primitive_function_num):
		active_function_sequence = genotype[0]
		input_table = genotype[1]
		output_table = genotype[2]
		input_function_table = genotype[3]
		input_len_table = genotype[4]
		active_function = genotype[5]
		function_instance_num = genotype[6]

		old_function = active_function[old_function_num]
		old_function_name = old_function.name
		old_primitive_function_name = old_function_name.split(":",1)[0]
		old_primitive_function_num = None
		for i in range(len(self.primitive_function)):
			if old_primitive_function_name == self.primitive_function[i][0]:
				old_primitive_function_num = i
				break
		old_function_instance_num = int(old_function_name.split(":",1)[1])

		#new_primitive_function_num = np.random.randint(len(self.primitive_function))
		#new_primitive_function_num = np.random.choice(len(self.primitive_function), 1,
		#		p=self.primitive_function_distribution)[0]

		new_function_instance_num = self.find_missing(function_instance_num[new_primitive_function_num])
		func_name = self.primitive_function[new_primitive_function_num][0] + ':' +  str(new_function_instance_num)

		new_function = Function(func_name, old_function.input_size,
			*self.primitive_function[new_primitive_function_num][1])

		new_function_name = new_function.name

		if new_function.input_size > old_function.input_size:
			new_input_function_table = copy.deepcopy(input_function_table[old_function_num])
			new_input_len_table = copy.deepcopy(input_len_table[old_function_num])
			new_input_table = [None] * new_function.input_size
			idx = np.random.randint(new_function.input_size - old_function.input_size + 1)
			new_input_table[idx:idx+old_function.input_size] = input_table[old_function_num]

			left_none_len = 0
			if new_input_table[0] == None:
				for i in range(len(new_input_table)):
					if new_input_table[i] == None:
						left_none_len += 1
					else:
						break

			if left_none_len > 0:
				input_table[old_function_num][0:0] = [None] * left_none_len
				new_input_function_candidate = [-1] + range(old_function_num)
				new_input_function_list = np.random.choice(new_input_function_candidate, left_none_len)
				new_new_input_function_table = []
				new_new_input_len_table = []

				idx = 0
				remained_len = left_none_len
				for i in range(len(new_input_function_list)):
					if new_input_function_list[i] == -1:
						new_input_function_name = 'input'
						new_input_function_output_size = self.input_size
					else:
						new_input_function_name = active_function_sequence[new_input_function_list[i]]
						new_input_function = active_function[new_input_function_list[i]]
						new_input_function_output_size = new_input_function.output_size

					if new_input_function_list[i] == -1:
						new_input = range(self.input_size)
					else:
						new_input = output_table[new_input_function_list[i]]

					if remained_len >= len(new_input):
						input_table[old_function_num][idx:idx+new_input_function_output_size] = new_input
						remained_len -= len(new_input)
						idx += len(new_input)
						new_new_input_len_table.append(len(new_input))
					else:
						input_idx = np.random.randint(len(new_input)-remained_len+1)
						input_table[old_function_num][idx:idx+remained_len] = new_input[input_idx:input_idx+remained_len]
						new_new_input_len_table.append(remained_len)
						remained_len = 0
						idx += remained_len 

					new_new_input_function_table.append(new_input_function_list[i])

					if remained_len == 0:
						break

				input_function_table[old_function_num][0:0] = new_new_input_function_table
				input_len_table[old_function_num][0:0] = new_new_input_len_table

			right_none_len = 0
			if new_input_table[-1] == None:
				for i in range(len(new_input_table)-1, 0, -1):
					if new_input_table[i] == None:
						right_none_len += 1
					else:
						break

			if right_none_len > 0:
				input_table[old_function_num].extend([None] * right_none_len)

				new_input_function_candidate = [-1] + range(old_function_num)
				new_input_function_list = np.random.choice(new_input_function_candidate, right_none_len)
				new_new_input_function_table = []
				new_new_input_len_table = []

				idx = len(new_input_table) - right_none_len
				remained_len = right_none_len
				for i in range(len(new_input_function_list)):
					if new_input_function_list[i] == -1:
						new_input_function_name = 'input'
						new_input_function_output_size = self.input_size
					else:
						new_input_function_name = active_function_sequence[new_input_function_list[i]]
						new_input_function = active_function[new_input_function_list[i]]
						new_input_function_output_size = new_input_function.output_size

					if new_input_function_list[i] == -1:
						new_input = range(self.input_size)
					else:
						new_input = output_table[new_input_function_list[i]]

					if remained_len >= len(new_input):
						input_table[old_function_num][idx:idx+new_input_function_output_size] = new_input
						remained_len -= len(new_input)
						idx += len(new_input)
						new_new_input_len_table.append(len(new_input))
					else:
						input_idx = np.random.randint(len(new_input)-remained_len+1)
						input_table[old_function_num][idx:idx+remained_len] = new_input[input_idx:input_idx+remained_len]
						new_new_input_len_table.append(remained_len)
						remained_len = 0
						idx += remained_len 

					new_new_input_function_table.append(new_input_function_list[i])

					if remained_len == 0:
						break

				input_function_table[old_function_num].extend(new_new_input_function_table)
				input_len_table[old_function_num].extend(new_new_input_len_table)

		elif new_function.input_size < old_function.input_size:
			new_start_idx = np.random.randint(old_function.input_size - new_function.input_size + 1)
			input_table[old_function_num] = input_table[old_function_num][new_start_idx:new_start_idx+new_function.input_size]

			del input_function_table[old_function_num][:]
			del input_len_table[old_function_num][:]
			previous = -100
			previous_function_num = None
			input_len = 0
			for i in range(len(input_table[old_function_num])):
				current = input_table[old_function_num][i]
				current_function_num = self.find_function_num_by_input(genotype, current)
				if current == previous + 1 and current_function_num == previous_function_num:
					input_len += 1
				else:
					if input_len != 0:
						input_len_table[old_function_num].append(input_len)

					input_function_table[old_function_num].append(current_function_num)
					input_len = 1

				previous = current
				previous_function_num = current_function_num
			input_len_table[old_function_num].append(input_len)

		active_function_sequence[old_function_num] = new_function_name
		active_function[old_function_num] = new_function
		function_instance_num[old_primitive_function_num].remove(old_function_instance_num)
		function_instance_num[new_primitive_function_num].append(new_function_instance_num)
		function_instance_num[new_primitive_function_num].sort()

		if new_function.output_size > old_function.output_size:
			original_output = copy.deepcopy(output_table[old_function_num])
			increase = new_function.output_size-old_function.output_size
			output_table[old_function_num].extend(range(output_table[old_function_num][-1]+1,
				output_table[old_function_num][-1]+1+increase))

			for i in range(old_function_num+1, len(output_table)):
				for j in range(len(output_table[i])):
					if output_table[i][j] > original_output[-1]:
						output_table[i][j] += increase

			for i in range(old_function_num+1, len(input_table)):
				for j in range(len(input_table[i])):
					if input_table[i][j] > original_output[-1]:
						input_table[i][j] += increase

			random_incrase_for_input = np.random.randint(increase+1)
			for i in range(old_function_num+1, len(input_table)):
				for j in range(len(input_table[i])):
					if min(output_table[old_function_num]) <= input_table[i][j] and \
						input_table[i][j] <= max(output_table[old_function_num]):
							input_table[i][j] += random_incrase_for_input

		elif new_function.output_size < old_function.output_size:

			original_output = copy.deepcopy(output_table[old_function_num])
			decrease = old_function.output_size-new_function.output_size
			start_idx = np.random.randint(decrease + 1)
			end_idx = start_idx + new_function.output_size
			survived_output = original_output[start_idx:end_idx]

			output_table[old_function_num] = original_output[0:new_function.output_size]

			for i in range(old_function_num+1, len(output_table)):
				for j in range(len(output_table[i])):
					if output_table[i][j] > original_output[-1]:
						output_table[i][j] -= decrease

			for i in range(old_function_num+1, len(input_table)):
				for j in range(len(input_table[i])):
					if input_table[i][j] in original_output:
						if input_table[i][j] in survived_output:
							input_table[i][j] -= start_idx
						else:
							input_table[i][j] = None
					elif input_table[i][j] > original_output[-1]:
						input_table[i][j] -= decrease

			none_input_list = []
			for i in range(old_function_num+1, len(input_table)):
				start = None
				end = None
				for j in range(len(input_table[i])):
					if input_table[i][j] == None:
						if start == None:
							start = j
					else:
						if start != None:
							end = j-1
							#print('i, start, end', i, start, end)
							none_input_list.append([i, start, end])
							start = None
							end = None
				if start != None and end == None:
					end = j
					none_input_list.append([i, start, end])

			for i in range(len(none_input_list)):
				target_function_num = none_input_list[i][0]
				start_pos = none_input_list[i][1]
				end_idx = none_input_list[i][2]
				input_len = end_idx - start_pos + 1

				if target_function_num == len(input_table) - 1:
					current = set(input_table[-1])
					current.remove(None)
					candidate = set(range(output_table[-1][-1]+1)).difference(current)
					candidate = list(candidate)
					np.random.shuffle(candidate)
					for j in range(input_len):
						input_table[target_function_num][start_pos+j] = candidate[j]
					continue

				new_input_function_candidate_num = range(-1, target_function_num)
				np.random.shuffle(new_input_function_candidate_num)

				remained_input_size = input_len

				for new_input_function_num in new_input_function_candidate_num:
					new_input_function_name = active_function_sequence[new_input_function_num]
					new_input_function = active_function[new_input_function_num]
					if new_input_function_num == -1:
						available = self.input_size
					else:
						available = new_input_function.output_size

					if available >= remained_input_size:
						start = np.random.randint(available - remained_input_size + 1)
						if new_input_function_num == -1:
							input_table[target_function_num][start_pos:start_pos+remained_input_size] = \
								range(start, start+remained_input_size)
						else:
							input_table[target_function_num][start_pos:start_pos+remained_input_size] = \
								output_table[new_input_function_num][start:start+remained_input_size]
						start_pos += remained_input_size
						remained_input_size = 0
					else:
						start = 0
						if new_input_function_num == -1:
							input_table[target_function_num][start_pos:start_pos+available] = \
								range(0, available)
						else:
							input_table[target_function_num][start_pos:start_pos+available] = \
								output_table[new_input_function_num][start:start+available]
						start_pos += available
						remained_input_size -= available

					if remained_input_size == 0:
						break

				if remained_input_size != 0:
					raise Exception('remained_input_size != 0')
					return None

			for i in range(len(none_input_list)):
				target_function_num = none_input_list[i][0]
				start_pos = none_input_list[i][1]
				end_idx = none_input_list[i][2]
				input_len = end_idx - start_pos + 1

				if target_function_num == len(input_table) - 1:
					continue

				current = None
				previous = None
				current_function_num = None
				previous_function_num = None
				input_len = 1
				input_function_list = []
				input_len_list = []
				for j in input_table[target_function_num]:
					current = j
					current_function_num = self.find_function_num_by_input(genotype, current)

					if previous != None:
						if current == previous + 1 and current_function_num == previous_function_num:
							input_len += 1
						else:
							input_len_list.append(input_len)
							input_len = 1

					previous = current
					previous_function_num = current_function_num
				input_len_list.append(input_len)

				idx = -1
				for j in input_len_list:
					idx += j

					if input_table[target_function_num][idx] < self.input_size:
						input_function_num = -1
					else:
						for k in range(len(output_table)):
							if input_table[target_function_num][idx] <= output_table[k][-1]:
								input_function_num = k
								break

					input_function_list.append(input_function_num)

				input_function_table[target_function_num] = input_function_list
				input_len_table[target_function_num] = input_len_list

			output_input_set = []
			for i in input_table[-1]:
				input_function_num = 0
				if i < self.input_size:
					input_function_num = -1
				else:
					for j in range(len(output_table)):
						if i <= output_table[j][-1]:
							input_function_num = j
							break
				output_input_set.append(input_function_num)

			output_input_list = list(set(output_input_set))
			output_input_list.sort()
			input_function_table[-1] = output_input_list

		return genotype

	def replace_output(self, genotype, gene, to_mutate):
		active_function_sequence = genotype[0]
		input_table = genotype[1]
		output_table = genotype[2]
		input_function_table = genotype[3]
		input_len_table = genotype[4]

		idx_to_change = np.random.randint(len(input_table[-1]))
		candidate = set(range(output_table[-1][-1]+1)).difference(input_table[-1])

		old_value = input_table[-1][idx_to_change]

		if old_value < self.input_size:
			old_function_num = -1
		else:
			for i in range(len(output_table)):
				if old_value <= output_table[i][-1]:
					old_function_num = i
					break

		new_value = np.random.choice(list(candidate))

		if new_value < self.input_size:
			new_function_num = -1
		else:
			for i in range(len(output_table)):
				if new_value <= output_table[i][-1]:
					new_function_num = i
					break

		found = False
		if new_function_num == -1:
			for i in input_table[-1]:
				if  i < self.input_size:
					found = True
					break
		else:
			for i in input_table[-1]:
				if  output_table[new_function_num][0] <= i and \
					i <= output_table[new_function_num][-1]:
						found = True
						break

		if found == False:
			input_function_table[-1].append(new_function_num)
			input_function_table[-1].sort()

		input_table[-1][idx_to_change] = new_value

		found = False
		if old_function_num == -1:
			for i in input_table[-1]:
				if  i < self.input_size:
					found = True
					break
		else:
			for i in input_table[-1]:
				if  output_table[old_function_num][0] <= i and \
					i <= output_table[old_function_num][-1]:
						found = True
						break

		if found == False:
			input_function_table[-1].remove(old_function_num)

		return genotype

	def remove_input_function(self, genotype, gene, to_mutate):

		active_function_sequence = genotype[0]
		input_table = genotype[1]
		output_table = genotype[2]
		input_function_table = genotype[3]
		input_len_table = genotype[4]
		active_function = genotype[5]
		function_instance_num = genotype[6]

		input_function_num = None
		input_function_idx = None
		input_sum = 0
		for i in range(len(input_function_table)):
			for j in range(len(input_function_table[i])):
				input_sum += 1
				if input_sum == to_mutate - len(active_function_sequence)+1:
					input_function_num = i
					input_function_idx = j
					break

			if input_function_num and input_function_idx:
				break

		target_function_name = active_function_sequence[input_function_num]
		target_function = active_function[input_function_num]

		if len(input_function_table[input_function_num]) <= 1:
			return None

		to_remove_len = input_len_table[input_function_num][input_function_idx]

		idx = 0
		for i in range(input_function_idx):
			idx += input_len_table[input_function_num][i]

		if input_function_idx < len(input_function_table[input_function_num]) - 1:
		#if input_function_idx == 0:
			expand_function_idx = input_function_idx + 1
			expand_input_idx = 0
			for i in range(input_function_idx+1):
				expand_input_idx += input_len_table[input_function_num][i]

			expand_function_num = input_function_table[input_function_num][expand_function_idx]
			if  expand_function_num == -1:
				available = input_table[input_function_num][expand_input_idx]
			else:
				available = input_table[input_function_num][expand_input_idx] - output_table[expand_function_num][0]

			if available >= to_remove_len:
				input_table[input_function_num][idx:idx+to_remove_len] \
					= range(input_table[input_function_num][expand_input_idx]-to_remove_len,
						input_table[input_function_num][expand_input_idx])
				del input_function_table[input_function_num][input_function_idx]
				input_len_table[input_function_num][expand_function_idx] += to_remove_len
				del input_len_table[input_function_num][input_function_idx]
			else:
				return None

		#elif input_function_idx < len(input_function_table[input_function_num]) - 1:
		#	print('remove a middle connection')
		else:
			"""
			print('remove a last connection')
			expand_function_idx = input_function_idx - 1
			print('expand_function_idx', expand_function_idx)
			expand_input_idx = 0
			for i in range(input_function_idx-1):
				expand_input_idx += input_len_table[input_function_num][i]
			print('expand_input_idx', expand_input_idx)

			expand_function_num = input_function_table[input_function_num][expand_function_idx]
			print('expand_function_num', expand_function_num)
			if  expand_function_num == -1:
				available = self.input_size - input_table[input_function_num][expand_input_idx] - input_len_table[input_function_num][expand_input_idx]
			else:
				available = output_table[expand_function_num][-1] - input_table[input_function_num][expand_input_idx] - input_len_table[input_function_num][expand_function_idx] + 1
			print('available', available)
			"""
			return None # no support for removing a last connection

		return genotype

	def add_input_function(self, genotype, gene, to_mutate):

		active_function_sequence = genotype[0]
		input_table = genotype[1]
		output_table = genotype[2]
		input_function_table = genotype[3]
		input_len_table = genotype[4]
		active_function = genotype[5]

		input_function_num = None
		input_function_idx = None
		input_sum = 0
		for i in range(len(input_function_table)):
			for j in range(len(input_function_table[i])):
				input_sum += 1
				if input_sum == to_mutate - len(active_function_sequence)+1:
					input_function_num = i
					input_function_idx = j
					break

			if input_function_num and input_function_idx:
				break

		target_function_name = active_function_sequence[input_function_num]
		target_function = active_function[input_function_num]

		new_input_function_candidate = range(-1, input_function_num)

		new_input_function_num = np.random.choice(new_input_function_candidate)

		if new_input_function_num == -1:
			new_input_function_name = 'input'
			input_add_max_len = self.input_size
		else:
			new_input_function_name = active_function_sequence[new_input_function_num]
			#new_input_function = self.function_dict[new_input_function_name]
			new_input_function = active_function[new_input_function_num]
			input_add_max_len = new_input_function.output_size

		if input_add_max_len >= input_len_table[input_function_num][input_function_idx]:
			input_add_max_len = input_len_table[input_function_num][input_function_idx] - 1
		if input_add_max_len <= 0:
			return None

		input_add_len = np.random.randint(input_add_max_len) + 1

		idx = 0
		for i in range(input_function_idx):
			idx += input_len_table[input_function_num][i]

		# front or rear?
		is_front = True
		if np.random.randint(2) == 0:
			is_front = False

		if is_front is False:
			idx += input_len_table[input_function_num][input_function_idx] - input_add_len

		if new_input_function_num == -1:
			input_add_idx = np.random.randint(self.input_size - input_add_len + 1)
			input_add = range(self.input_size)[input_add_idx:input_add_idx+input_add_len]
		else:
			input_add_idx = np.random.randint(len(output_table[new_input_function_num]) - input_add_len + 1)
			input_add = output_table[new_input_function_num][input_add_idx:input_add_idx+input_add_len]

		input_table[input_function_num][idx:idx+input_add_len] = input_add

		if is_front is True:
			input_function_table[input_function_num].insert(input_function_idx, new_input_function_num)
			input_len_table[input_function_num][input_function_idx] -= input_add_len
			input_len_table[input_function_num].insert(input_function_idx, input_add_len)
		else:
			input_function_table[input_function_num].insert(input_function_idx+1, new_input_function_num)
			input_len_table[input_function_num][input_function_idx] -= input_add_len
			input_len_table[input_function_num].insert(input_function_idx+1, input_add_len)

		return genotype

	def replace_input_function(self, genotype, gene, to_mutate):

		active_function_sequence = genotype[0]
		input_table = genotype[1]
		output_table = genotype[2]
		input_function_table = genotype[3]
		input_len_table = genotype[4]
		active_function = genotype[5]

		input_function_num = None
		input_function_idx = None
		input_sum = 0
		for i in range(len(input_function_table)):
			for j in range(len(input_function_table[i])):
				input_sum += 1
				if input_sum == to_mutate - len(active_function_sequence)+1:
					input_function_num = i
					input_function_idx = j
					break

			if input_function_num and input_function_idx:
				break

		target_function_name = active_function_sequence[input_function_num]
		target_function = active_function[input_function_num]

		new_input_function_candidate = []
		if self.input_size >= target_function.input_size:
			new_input_function_candidate.append(-1)

		input_size = input_len_table[input_function_num][input_function_idx]
		for i in range(len(active_function_sequence)):
			function = active_function[i]
			#if function.output_size >= target_function.input_size:
			if function.output_size >= input_size:
				new_input_function_candidate.append(i)

		if len(new_input_function_candidate) == 0:
			return None

		new_input_function_candidate = [i for i in new_input_function_candidate if i < input_function_num]
		#new_input_function_candidate.remove(input_function_table[input_function_num][input_function_idx])

		if len(new_input_function_candidate) == 0:
			return None

		new_input_function_num = np.random.choice(new_input_function_candidate)

		if new_input_function_num == -1:
			new_input_max_len = self.input_size
		else:
			new_input_function_name = active_function_sequence[new_input_function_num]
			new_input_function = active_function[new_input_function_num]
			new_input_max_len = new_input_function.output_size

		new_input_idx = np.random.randint(new_input_max_len - input_size+1)
		if new_input_function_num != -1:
			new_input_idx += output_table[new_input_function_num][0]

		idx = 0
		for i in range(input_function_idx):
			idx += input_len_table[input_function_num][i]

		input_function_table[input_function_num][input_function_idx] = new_input_function_num
		input_table[input_function_num][idx:idx+input_len_table[input_function_num][input_function_idx]] = \
			range(new_input_idx, new_input_idx+input_len_table[input_function_num][input_function_idx])

		return genotype

	def find_missing(self, lst):
		if not lst:
			return 0
		missing = [x for x in range(0, lst[-1]+2) if x not in lst]
		return missing[0]

	def generate_genotype(self):
		active_function = []
		function_instance_num = [[] for _ in range(len(self.primitive_function))]
		active_function_sequence = []
		for i in range(self.row_size * self.column_size):
			#idx = np.random.randint(len(self.primitive_function))
			idx = np.random.choice(len(self.primitive_function), 1,
				p=self.primitive_function_distribution)[0]
			func_idx = self.find_missing(function_instance_num[idx])
			function_instance_num[idx].append(func_idx)
			function_instance_num[idx].sort()
			func_name = self.primitive_function[idx][0] + ':' +  str(func_idx)
			active_function_sequence.append(func_name)

			func = Function(func_name, np.random.randint(low=1,
				high=(self.input_size+1)),
				*self.primitive_function[idx][1])
			active_function.append(func)

		output_table = []
		input_table = []
		input_function_table = []
		input_len_table = []

		current_output_num = self.input_size

		for i in range(len(active_function_sequence)):
			function = active_function[i]
			function_output_sequence = range(current_output_num,
				current_output_num+function.output_size)
			output_table.append(function_output_sequence)
			current_output_num += function.output_size
			function_input_sequence = []
			remained_input_size = function.input_size
			parent_list = []
			num_of_parent = 0
			input_len = []

			if i == 0:
				start_num = 0
				if function.input_size > self.input_size:
					return None
				elif function.input_size == self.input_size:
					start_num = 0
				else:
					start_num = np.random.randint(self.input_size - function.input_size,
						size=1)
				end_num = start_num + function.input_size
				function_input_sequence += range(start_num, end_num)
				parent_list = [-1]
				num_of_parent += 1
				input_len.append(function.input_size)
			else:
				input_function_num = np.random.choice(i+1, function.input_size, replace=True) - 1

				for j in input_function_num:
					start_num = 0
					end_num = 0
					input_size = self.input_size

					if j != -1:
						start_num = output_table[j][0]
						input_size = active_function[j].output_size

					if remained_input_size >= input_size:
						end_num = start_num + input_size
						remained_input_size -= input_size
					else:
						start_num += np.random.randint(input_size - remained_input_size,
							size=1)
						end_num = start_num + remained_input_size
						remained_input_size = 0 

					function_input_sequence += range(start_num, end_num)
					num_of_parent += 1
					input_len.append(int(end_num - start_num))

					if remained_input_size == 0:
						break

				parent_list = list(input_function_num[0:num_of_parent])
				if remained_input_size != 0:
					raise Exception('remained_input_size != 0') 
					return None

			input_function_table.append(parent_list)
			input_table.append(function_input_sequence)
			input_len_table.append(input_len)

		# skip sanity check of the generated genotype

		input_for_output = list(np.random.choice(output_table[-1][-1],
			self.output_size, replace=False))
		input_table.append(input_for_output)

		is_input_used = None
		parent_list_for_output = [None] * len(active_function_sequence)
		for input_ in input_for_output:
			if input_ < self.input_size:
				is_input_used = 1

			for i in range(len(output_table)): 
				if input_ in output_table[i]:
					parent_list_for_output[i] = 1
					break

			if all(parent == 1 for parent in parent_list_for_output):
				break

		parent_list_for_output = [is_input_used] + parent_list_for_output

		parent_list = [i-1 for i, e in enumerate(parent_list_for_output) if e != None]

		input_function_table.append(parent_list)

		return active_function_sequence, input_table, output_table, input_function_table, \
			input_len_table, active_function, function_instance_num
	
	def register_primitive_function(self, function):
		self.primitive_function.append(function)

	def register_primitive_function_set(self, function_set):
		self.primitive_function = function_set

	def register_function_set(self, primitive_functions, function_idx):
		identity_idx = -1
		previous_f_idx = -1
		for idx in function_idx:
			if previous_f_idx != idx:
				identity_idx = 0
			else:
				identity_idx += 1

			func = Function(primitive_functions[idx][0] + str(identity_idx),
				np.random.randint(low=1, high=(self.input_size+1)//1),
				*primitive_functions[idx][1])
			self.register_function(func)
			previous_f_idx = idx

	def neuron_population(self, individual, function_no):
		if self.use_neuron_function == 0:
			return 0

		if individual.valid_function[function_no] == 0:
			return 0

		neuron_function_name = self.primitive_function[-1][0]
		active_function_sequence = individual.genotype[5]
		neuron_function = active_function_sequence[function_no]

		neuron_num = 0
		if neuron_function_name in neuron_function.name:
			neuron_num += neuron_function.parameter['input_size']
			neuron_num += neuron_function.parameter['hidden_layer_size']
			neuron_num += neuron_function.parameter['output_size']
			neuron_num += neuron_function.parameter['weight'].size

		return neuron_num

	def compute_loss(self, individual, input_data, output_data):
		self.build_individual_network_tf(individual)
		graph = tf.get_default_graph()
		input_tensor = graph.get_tensor_by_name(self.output_base_name + '-1:0')
		output_tensor_ = graph.get_tensor_by_name('output_tensor_:0')
		loss = graph.get_tensor_by_name('loss:0')

		if input_data.ndim == 1:
			input_data = np.expand_dims(input_data, axis=0)

		iteration = input_data.shape[0] // self.tf_batch_size
		remained = input_data.shape[0] % self.tf_batch_size

		loss_sum = 0
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(iteration):
				loss_value = sess.run(loss,
					feed_dict={input_tensor:input_data[i*self.tf_batch_size:i*self.tf_batch_size+self.tf_batch_size],
						output_tensor_: output_data[i*self.tf_batch_size:i*self.tf_batch_size+self.tf_batch_size]})

				loss_sum += loss_value*self.tf_batch_size

			if remained > 0:
				loss_value = sess.run(loss,
					feed_dict={input_tensor:input_data[iteration*self.tf_batch_size:iteration*self.tf_batch_size+remained],
					output_tensor_:output_data[iteration*self.tf_batch_size:iteration*self.tf_batch_size+remained]})

				loss_sum += loss_value*remained

		tf.reset_default_graph()

		loss_avg = loss_sum / input_data.shape[0]
		return loss_avg
