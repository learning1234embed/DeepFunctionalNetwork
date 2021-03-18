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

def next_batch(input_data, output_data, batch_size):
	assert input_data.shape[0] == output_data.shape[0]
	data_num = np.random.choice(input_data.shape[0], size=batch_size,
		replace=False)
	batch_input_data = input_data[data_num,:]
	batch_output_data = output_data[data_num,:]
	return batch_input_data, batch_output_data

def interpretability_ratio(dfn, dnn_neuron_num, population_no):
	individual = dfn.population[population_no]
	valid_func_idx = np.where(individual.valid_function == 1)[0]
	valid_func_name = [individual.genotype[0][idx] for idx in valid_func_idx]
	print(valid_func_name)

	dfn_neuron_num = 0
	for i in range(len(individual.valid_function)):
		dfn_neuron_num += dfn.neuron_population(individual, i)
	interpretability = 1.0 - float(dfn_neuron_num) / float(dnn_neuron_num)
	return dfn_neuron_num, interpretability

def contribution_rate(dfn, input_data, output_data, iteration, batch_size,
	population_no):
	individual = dfn.population[population_no]
	active_function_sequence = individual.genotype[0]
	print(active_function_sequence)
	print(individual.valid_function)

	valid_func_idx = np.where(individual.valid_function == 1)[0]
	valid_func_name = [active_function_sequence[idx] for idx in valid_func_idx]
	print(valid_func_name)

	function_contribution_rate_list = []
	for i in range(iteration):
		print('%dth batch' % i)
		input_batch, output_batch = next_batch(input_data, output_data, batch_size)
		function_contribution_rate = \
			dfn.function_contribution_tf(individual,
			input_batch, output_batch, target='fitness')
		print(function_contribution_rate)
		function_contribution_rate_list.append(function_contribution_rate)

	function_contribution_rate = np.stack(function_contribution_rate_list)
	function_contribution_rate_sum = np.sum(function_contribution_rate, axis=0)
	return function_contribution_rate_sum

def main(args):
	if args.dnn_name == None or args.dnn_name == '':
		print('No dnn name. Use -dnn_name')
		return

	if not os.path.exists(args.dnn_name):
		print(args.dnn_name, 'does not exists')
		return

	dfn_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
		args.dnn_name), args.dfn_name)
	dfn_file_path = os.path.join(dfn_dir, args.dfn_name)
	print('dfn_file_path', dfn_file_path)

	if not os.path.exists(dfn_file_path):
		raise Exception('%s does not exists' % dfn_file_path)

	dfn = pickle.load(open(dfn_file_path, 'rb'))

	if args.mode == 'i':
		print('[i] interpretability ratio')

		if args.dnn_neuron_num <= -1:
			print('No dnn_neuron_num. Use -dnn_neuron_num')
			return

		dfn_neuron_num, interpretability = interpretability_ratio(dfn,
			args.dnn_neuron_num, args.population_no)
		print('dnn_neuron_num:', args.dnn_neuron_num)
		print('dfn_neuron_num:', dfn_neuron_num)
		print('interpretability_ratio:', interpretability)

		return

	elif args.mode == 'p':
		print('[p] performance gap')
		if args.input_data == None:
			print('[s] No input_data. Use -input_data')
			return

		if args.output_data == None:
			print('[s] No output_data. Use -output_data')
			return

		if not args.dnn_loss:
			print('No dnn_loss. Use -dnn_loss')
			return

		input_data = np.load(args.input_data)
		output_data = np.load(args.output_data)
		print('input_data.shape', input_data.shape)
		print('output_data.shape', output_data.shape)


		individual = dfn.population[args.population_no]
		valid_func_idx = np.where(individual.valid_function == 1)[0]
		valid_func_name = [individual.genotype[0][idx] for idx in valid_func_idx]
		print(valid_func_name)

		dfn_loss = dfn.compute_loss(individual, input_data, output_data)
		performance_gap = dfn_loss - args.dnn_loss
		print('dnn_loss:', args.dnn_loss)
		print('dfn_loss:', dfn_loss)
		print('performance_gap:', performance_gap)

		return

	elif args.mode == 'c':
		print('[c] contribution rate')

		if args.input_data == None:
			print('[s] No input_data. Use -input_data')
			return

		if args.output_data == None:
			print('[s] No output_data. Use -output_data')
			return

		input_data = np.load(args.input_data)
		output_data = np.load(args.output_data)
		print('input_data.shape', input_data.shape)
		print('output_data.shape', output_data.shape)

		function_contribution_rate_sum = contribution_rate(dfn, input_data, output_data,
			args.contribution_iteration, args.contribution_batch_size)
		print(function_contribution_rate_sum)
		return

	elif args.mode == 'f':
		print('[f] occurence frequency')
		individual = dfn.population[args.population_no]
		active_function_sequence = individual.genotype[0]

		raw_function_name_list = []
		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				raw_function_name = active_function_sequence[i].split(":",1)[0]
				raw_function_name_list.append(raw_function_name)

		raw_function_name_set = set(raw_function_name_list)
		function_name_list = list(raw_function_name_set)

		occurrence_frequency_list = []
		for function_name in function_name_list:
			occurrence_frequency = raw_function_name_list.count(function_name)
			occurrence_frequency_list.append(occurrence_frequency)

		print(function_name_list)
		print(occurrence_frequency_list)

		return

	elif args.mode == 'o':
		print('[o] iterpretability optimization')

		if args.target_interpretability < 0:
			print('[o] No target_interpretability. Use -target_interpretability')
			return

		if args.dnn_neuron_num <= -1:
			print('No dnn_neuron_num. Use -dnn_neuron_num')
			return

		if args.input_data == None:
			print('[o] No input_data. Use -input_data')
			return

		if args.output_data == None:
			print('[o] No output_data. Use -output_data')
			return

		input_data = np.load(args.input_data)
		output_data = np.load(args.output_data)
		print('input_data.shape', input_data.shape)
		print('output_data.shape', output_data.shape)

		val_input_data = input_data
		val_output_data = output_data

		if args.val_input_data and args.val_output_data:
			val_input_data = np.load(args.val_input_data)
			val_output_data = np.load(args.val_output_data)
			print('val_input_data', val_input_data.shape)
			print('val_output_data', val_output_data.shape)

		new_dfn_name = dfn.name + '_o'
		dfn.name = new_dfn_name
		new_dfn_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
			args.dnn_name), new_dfn_name)
		new_dfn_file_path = os.path.join(new_dfn_dir, new_dfn_name)
		print('new_dfn_file_path', new_dfn_file_path)

		individual = dfn.population[args.population_no]

		valid_func_idx = []
		neuron_population = []
		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				valid_func_idx.append(i)
				neuron_population.append(dfn.neuron_population(individual, i))
		n = np.asarray(neuron_population, dtype=np.float)
		print('valid_func_idx', valid_func_idx)
		print('neuron_population', neuron_population)

		contribution = contribution_rate(dfn, input_data, output_data,
			args.contribution_iteration, args.contribution_batch_size)
		c = np.asarray(contribution, dtype=np.float)
		print('contribution', contribution)

		ratio = n / c
		print('ratio', ratio)

		replace_idx_order = np.argsort(ratio)[::-1]
		print('replace_idx_order', replace_idx_order)

		ratio = np.sort(ratio)[::-1]
		print('sorted ratio', ratio)

		replace_func_idx_order = [valid_func_idx[i] for i in replace_idx_order]
		print('replace_func_idx_order', replace_func_idx_order)

		current_loss = dfn.compute_loss(individual, input_data, output_data)
		print('current_loss', current_loss)

		for i in range(len(replace_func_idx_order)):
			interpretability = interpretability_ratio(dfn,
				args.dnn_neuron_num, population_no)[1]
			print('interpretability', interpretability)
			if interpretability >= args.target_interpretability:
				break

			old_function_no = replace_func_idx_order[i]
			print('old_function_no', old_function_no)
			r = ratio[i]
			print('ratio', r)

			if r <= 0:
				continue

			best_loss = None
			best_individual = None
			for new_primitive_function_no in range(len(dfn.primitive_function)-1):
				individual_clone = copy.deepcopy(individual)
				new_genotype = dfn.replace_function(individual_clone.genotype,
					old_function_no, new_primitive_function_no)
				if not new_genotype:
					print('fail')
					continue

				individual_clone.genotype = new_genotype
				individual_clone.update_graph()
				individual_clone.update_valid_function()
				individual_clone.init_connection_weight(random_init=False)

				valid_func_idx = np.where(individual_clone.valid_function == 1)[0]
				valid_func_name = [individual_clone.genotype[0][idx] for idx in valid_func_idx]
				print('[%d] candidate valid_func_name' % new_primitive_function_no,
					valid_func_name)

				dfn.update_weight_tf(individual_clone, input_data, output_data,
					val_input_data, val_output_data,
					args.update_iteration, args.update_batch_size, save=1)

				loss = dfn.compute_loss(individual_clone, input_data, output_data)
				print('[%d] candidate loss' % new_primitive_function_no, loss)
				if not best_loss or loss < best_loss:
					best_loss = loss
					best_individual_clone = copy.deepcopy(individual_clone)

				valid_func_idx = np.where(best_individual_clone.valid_function == 1)[0]
				valid_func_name = [best_individual_clone.genotype[0][idx] for idx in valid_func_idx]
				print('[%d] best_individual valid_func_name' % new_primitive_function_no,
					valid_func_name)
				print('[%d] best_loss' % new_primitive_function_no, best_loss)

			individual = dfn.population[args.population_no] = best_individual_clone

		valid_func_idx = np.where(individual.valid_function == 1)[0]
		valid_func_name = [individual.genotype[0][idx] for idx in valid_func_idx]
		print('final individual valid_func_name', valid_func_name)

		if not os.path.exists(new_dfn_dir):
			os.makedirs(new_dfn_dir)
		pickle.dump(dfn, open(new_dfn_file_path, 'wb'))

		return

	elif args.mode == 'l':
		print('[l] performance optimization')

		if args.input_data == None:
			print('[l] No input_data. Use -input_data')
			return

		if args.output_data == None:
			print('[l] No output_data. Use -output_data')
			return

		input_data = np.load(args.input_data)
		output_data = np.load(args.output_data)
		print('input_data.shape', input_data.shape)
		print('output_data.shape', output_data.shape)

		val_input_data = input_data
		val_output_data = output_data

		if args.val_input_data and args.val_output_data:
			val_input_data = np.load(args.val_input_data)
			val_output_data = np.load(args.val_output_data)
			print('val_input_data', val_input_data.shape)
			print('val_output_data', val_output_data.shape)

		new_dfn_name = dfn.name + '_l'
		dfn.name = new_dfn_name
		new_dfn_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
			args.dnn_name), new_dfn_name)
		new_dfn_file_path = os.path.join(new_dfn_dir, new_dfn_name)
		print('new_dfn_file_path', new_dfn_file_path)

		individual = dfn.population[args.population_no]

		valid_func_idx = []
		neuron_population = []
		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				valid_func_idx.append(i)
				neuron_population.append(dfn.neuron_population(individual, i))
		n = np.asarray(neuron_population, dtype=np.float)
		print('valid_func_idx', valid_func_idx)
		print('neuron_population', neuron_population)

		dfn.update_weight_tf(individual, input_data, output_data,
			val_input_data, val_output_data,
			args.update_iteration, args.update_batch_size, save=1)

		current_loss = dfn.compute_loss(individual, input_data, output_data)
		print('current_loss', current_loss)

		best_loss = current_loss
		best_individual_clone = individual

		contribution = contribution_rate(dfn, input_data, output_data,
			args.contribution_iteration, args.contribution_batch_size)
		c = np.asarray(contribution, dtype=np.float)
		print('contribution', contribution)

		replace_idx_order = np.argsort(c)[::-1]
		print('replace_idx_order', replace_idx_order)

		replace_func_idx_order = [valid_func_idx[i] for i in replace_idx_order]
		print('replace_func_idx_order', replace_func_idx_order)

		for i in range(len(replace_func_idx_order)):
			interpretability = interpretability_ratio(dfn,
				args.dnn_neuron_num, population_no)[1]
			print('interpretability', interpretability)

			old_function_no = replace_func_idx_order[i]
			print('old_function_no', old_function_no)

			#best_loss = None
			#best_individual = None
			for new_primitive_function_no in range(len(dfn.primitive_function)):
				individual_clone = copy.deepcopy(individual)
				new_genotype = dfn.replace_function(individual_clone.genotype,
					old_function_no, new_primitive_function_no)
				if not new_genotype:
					print('fail')
					continue

				individual_clone.genotype = new_genotype
				individual_clone.update_graph()
				individual_clone.update_valid_function()
				individual_clone.init_connection_weight(random_init=False)

				valid_func_idx = np.where(individual_clone.valid_function == 1)[0]
				valid_func_name = [individual_clone.genotype[0][idx] for idx in valid_func_idx]
				print('[%d] candidate valid_func_name' % new_primitive_function_no,
					valid_func_name)

				dfn.update_weight_tf(individual_clone, input_data, output_data,
					val_input_data, val_output_data,
					args.update_iteration, args.update_batch_size, save=1)

				loss = dfn.compute_loss(individual_clone, input_data, output_data)
				print('[%d] candidate loss' % new_primitive_function_no, loss)
				if not best_loss or loss < best_loss:
					best_loss = loss
					best_individual_clone = copy.deepcopy(individual_clone)

				valid_func_idx = np.where(best_individual_clone.valid_function == 1)[0]
				valid_func_name = [best_individual_clone.genotype[0][idx] for idx in valid_func_idx]
				print('[%d] best_individual valid_func_name' % new_primitive_function_no,
					valid_func_name)
				print('[%d] best_loss' % new_primitive_function_no, best_loss)

			individual = dfn.population[args.population_no] = best_individual_clone

		valid_func_idx = np.where(individual.valid_function == 1)[0]
		valid_func_name = [individual.genotype[0][idx] for idx in valid_func_idx]
		print('final individual valid_func_name', valid_func_name)

		if not os.path.exists(new_dfn_dir):
			os.makedirs(new_dfn_dir)
		pickle.dump(dfn, open(new_dfn_file_path, 'wb'))

		return

	elif args.mode == 'n':
		print('[n] replace a non-neuron function with a neuron function')

		if args.input_data == None:
			print('[n] No input_data. Use -input_data')
			return

		if args.output_data == None:
			print('[n] No output_data. Use -output_data')
			return

		input_data = np.load(args.input_data)
		output_data = np.load(args.output_data)
		print('input_data.shape', input_data.shape)
		print('output_data.shape', output_data.shape)

		val_input_data = input_data
		val_output_data = output_data

		if args.val_input_data and args.val_output_data:
			val_input_data = np.load(args.val_input_data)
			val_output_data = np.load(args.val_output_data)
			print('val_input_data', val_input_data.shape)
			print('val_output_data', val_output_data.shape)

		new_dfn_name = dfn.name + '_n'
		dfn.name = new_dfn_name
		new_dfn_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
			args.dnn_name), new_dfn_name)
		new_dfn_file_path = os.path.join(new_dfn_dir, new_dfn_name)
		print('new_dfn_file_path', new_dfn_file_path)

		individual = dfn.population[args.population_no]

		valid_func_idx = []
		neuron_population = []
		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				valid_func_idx.append(i)
				neuron_population.append(dfn.neuron_population(individual, i))
		n = np.asarray(neuron_population, dtype=np.float)
		print('valid_func_idx', valid_func_idx)
		print('neuron_population', neuron_population)

		dfn.update_weight_tf(individual, input_data, output_data,
			val_input_data, val_output_data,
			args.update_iteration, args.update_batch_size, save=1)

		current_loss = dfn.compute_loss(individual, input_data, output_data)
		print('current_loss', current_loss)

		best_loss = current_loss
		best_individual_clone = individual

		contribution = contribution_rate(dfn, input_data, output_data,
			args.contribution_iteration, args.contribution_batch_size)
		c = np.asarray(contribution, dtype=np.float)
		print('contribution', contribution)

		replace_idx_order = np.argsort(c)[::-1]
		print('replace_idx_order', replace_idx_order)

		replace_func_idx_order = [valid_func_idx[i] for i in replace_idx_order]
		print('replace_func_idx_order', replace_func_idx_order)

		for i in range(len(replace_func_idx_order)):
			interpretability = interpretability_ratio(dfn,
				args.dnn_neuron_num, population_no)[1]
			print('interpretability', interpretability)

			old_function_no = replace_func_idx_order[i]
			print('old_function_no', old_function_no)
			print('contribution', c[old_function_no])

			n = dfn.neuron_population(individual, old_function_no)
			print('n', n)
			if n > 0:
				continue

			new_primitive_function_no = len(dfn.primitive_function)-1
			individual_clone = copy.deepcopy(individual)
			new_genotype = dfn.replace_function(individual_clone.genotype,
				old_function_no, new_primitive_function_no)
			if not new_genotype:
				print('fail')
				continue

			individual_clone.genotype = new_genotype
			individual_clone.update_graph()
			individual_clone.update_valid_function()
			individual_clone.init_connection_weight(random_init=False)

			valid_func_idx = np.where(individual_clone.valid_function == 1)[0]
			valid_func_name = [individual_clone.genotype[0][idx] for idx in valid_func_idx]
			print('[%d] candidate valid_func_name' % new_primitive_function_no,
				valid_func_name)

			dfn.update_weight_tf(individual_clone, input_data, output_data,
				val_input_data, val_output_data,
				args.update_iteration, args.update_batch_size, save=1)

			loss = dfn.compute_loss(individual_clone, input_data, output_data)
			print('[%d] candidate loss' % new_primitive_function_no, loss)
			if loss < best_loss or np.isnan(best_loss):
				best_loss = loss
				best_individual_clone = copy.deepcopy(individual_clone)

			valid_func_idx = np.where(best_individual_clone.valid_function == 1)[0]
			valid_func_name = [best_individual_clone.genotype[0][idx] for idx in valid_func_idx]
			print('[%d] best_individual valid_func_name' % new_primitive_function_no,
				valid_func_name)
			print('[%d] best_loss' % new_primitive_function_no, best_loss)

			individual = dfn.population[args.population_no] = best_individual_clone

		valid_func_idx = np.where(individual.valid_function == 1)[0]
		valid_func_name = [individual.genotype[0][idx] for idx in valid_func_idx]
		print('final individual valid_func_name', valid_func_name)

		if not os.path.exists(new_dfn_dir):
			os.makedirs(new_dfn_dir)
		pickle.dump(dfn, open(new_dfn_file_path, 'wb'))

		return

	elif args.mode == 'r':
		print('[r] function replacement')

		if args.old_function_no < 0:
			print('[o] No old_function. Use -old_function')
			return

		if args.new_function_no < 0:
			print('[o] No new_function. Use -new_function')
			return

		new_dfn_name = dfn.name + '_r'
		dfn.name = new_dfn_name
		new_dfn_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
			args.dnn_name), new_dfn_name)
		new_dfn_file_path = os.path.join(new_dfn_dir, new_dfn_name)
		print('new_dfn_file_path', new_dfn_file_path)

		individual = dfn.population[args.population_no]

		valid_func_idx = []
		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				valid_func_idx.append(i)
				print('[%d] %s' % (i, individual.genotype[0][i]))

		print('old_function[%d]: %s' % (args.old_function_no,
			individual.genotype[0][args.old_function_no]))
		#print('new_function[%d]: %s' % (args.new_function_no, individual.genotype[0][args.new_function_no]))

		new_genotype = dfn.replace_function(individual.genotype,
			args.old_function_no, args.new_function_no)
		if not new_genotype:
			print('function replacement fail')
			return

		print('function replacement succeed')
		individual.genotype = new_genotype
		individual.update_graph()
		individual.update_valid_function()
		individual.init_connection_weight(random_init=False)

		valid_func_idx = []
		for i in range(len(individual.valid_function)):
			if individual.valid_function[i] == 1:
				valid_func_idx.append(i)
				print('[%d] %s' % (i, individual.genotype[0][i]))

		if not os.path.exists(new_dfn_dir):
			os.makedirs(new_dfn_dir)
		pickle.dump(dfn, open(new_dfn_file_path, 'wb'))

		return

	#if not os.path.exists(dfn_dir):
	#	os.makedirs(dfn_dir)
	#pickle.dump(dfn, open(dfn_file_path, 'wb'))

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('-dnn_name', type=str, help='dnn_name', default=None)
	parser.add_argument('-dfn_name', type=str, help='dfn_name', default=None)
	parser.add_argument('-mode', type=str,	help='mode', default=None)
	parser.add_argument('-dnn_neuron_num', type=int, help='dnn_neuron_num', default=-1)
	parser.add_argument('-dnn_loss', type=float, help='dnn_loss', default=None)
	parser.add_argument('-target_interpretability', type=float, help='target_interpretability', default=-1.0)
	parser.add_argument('-input_data', type=str, help='input_data', default=None)
	parser.add_argument('-output_data', type=str, help='output_data', default=None)
	parser.add_argument('-val_input_data', type=str, help='val_input_data', default=None)
	parser.add_argument('-val_output_data', type=str, help='val_output_data', default=None)
	parser.add_argument('-contribution_batch_size', type=int, help='contribution_batch_size', default=1)
	parser.add_argument('-contribution_iteration', type=int, help='contribution_iteration', default=100)
	parser.add_argument('-update_iteration', type=int, help='update_iteration', default=10000)
	parser.add_argument('-update_batch_size', type=int, help='update_batch_size', default=100)
	parser.add_argument('-population_no', type=int,	help='population_no', default=0)
	parser.add_argument('-old_function_no', type=int, help='old_function_no', default=-1)
	parser.add_argument('-new_function_no', type=int, help='new_function_no', default=-1)

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
