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
from function_estimator import FunctionEstimator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
#np.set_printoptions(threshold=np.nan)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

def main(args):
	np.random.seed(args.seed)

	if args.dnn_name == None or args.dnn_name == '':
		print('No dnn name. Use -dnn_name')
		return

	if not os.path.exists(args.dnn_name):
		print(args.dnn_name, 'does not exists')
		return

	if not args.dfn_name:
		print('No dfn name. Use -dfn_name')
		return

	dfn_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
		args.dnn_name), args.dfn_name)
	dfn_file_path = os.path.join(dfn_dir, args.dfn_name)
	print('dfn_file_path', dfn_file_path)

	dfn = None
	if os.path.exists(dfn_file_path):
		dfn = pickle.load(open(dfn_file_path, 'rb'))

	if args.mode == 'c':
		print('[c] creating a deep functional network')
		if dfn:
			print(dfn_file_path, 'already exists')
			return

		if args.placeholder_size == -1:
			print('[c] No placeholder size. Use -placeholder_size')
			return

		if args.population_size == -1:
			print('[c] No population size. Use -population_size')
			return

		func_set = FunctionSet()

		if args.use_distribution == 0:
			if args.input_size == -1:
				print('[c] No input size. Use -input_size')
				return

			if args.output_size == -1:
				print('[c] No output size. Use -output_size')
				return

			input_size = args.input_size
			output_size = args.output_size
			function_distribution = None
		else:
			fe_dir = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
				args.dnn_name), 'function_estimator')
			fe_file_name = 'function_estimator.obj'
			fe_file_path = os.path.join(fe_dir, fe_file_name)
			print('fe_file_path', fe_file_path)

			fe = None
			if os.path.exists(fe_file_path):
				fe = pickle.load(open(fe_file_path, 'rb'))

			input_size = fe.input_size
			output_size = fe.output_size
			function_distribution = np.load(fe.distribution_file_path)

		#print('function_distribution')
		#print(function_distribution)
		primitive_idx = range(len(func_set.pool))
		#print('primitive_idx', primitive_idx)

		func_set.generate_primitive(primitive_idx, args.use_neuron_function)
		print('generating %d primitive functions' % (len(func_set.primitive)))

		print('generating a dfn')
		dfn = DFN(args.dfn_name, input_size, output_size, 1,
			args.placeholder_size, primitive_function=func_set.primitive,
			primitive_function_distribution=function_distribution,
			use_neuron_function=args.use_neuron_function,
			fitness=args.fitness, loss=args.loss)

		print('generating population of size %d' % (args.population_size))
		dfn.generate_population(args.population_size)

	elif args.mode == 's':
		print('[s] searching the dfn architecture')

		if not dfn:
			print(dfn_file_path, 'does not exists')
			return

		if args.input_data == None:
			print('[s] No input_data. Use -input_data')
			return

		if args.output_data == None:
			print('[s] No output_data. Use -output_data')
			return

		input_data = np.load(args.input_data)
		print('input_data', input_data.shape)
		output_data = np.load(args.output_data)
		print('output_data', output_data.shape)

		generation = 0
		for _ in range(args.generation // args.saving_cycle):
			print('[generation]', generation)
			dfn.run(args.saving_cycle, args.lamb, input_data, output_data)
			print('Saving', dfn_file_path)
			pickle.dump(dfn, open(dfn_file_path, 'wb'))
			generation += args.saving_cycle

		dfn.run(args.generation % args.saving_cycle, args.lamb,
			input_data, output_data)
		pickle.dump(dfn, open(dfn_file_path, 'wb'))

	elif args.mode == 'u':
		print('[u] update weight of the population[%d]' % args.population_no)

		if not dfn:
			print(dfn_file_path, 'does not exists')
			return

		if args.input_data == None:
			print('[u] No input_data. Use -input_data')
			return

		if args.output_data == None:
			print('[u] No output_data. Use -output_data')
			return

		input_data = np.load(args.input_data)
		output_data = np.load(args.output_data)
		print('input_data', input_data.shape)
		print('output_data', output_data.shape)

		val_input_data = input_data
		val_output_data = output_data

		if args.val_input_data and args.val_output_data:
			val_input_data = np.load(args.val_input_data)
			val_output_data = np.load(args.val_output_data)
			print('val_input_data', val_input_data.shape)
			print('val_output_data', val_output_data.shape)

		dfn.update_weight_tf(dfn.population[args.population_no],
			input_data, output_data,
			val_input_data, val_output_data,
			args.update_iteration, args.update_batch_size, save=args.save)

		if args.save == 1:
			print('Saving', dfn_file_path)
			pickle.dump(dfn, open(dfn_file_path, 'wb'))
		return

	elif args.mode == 'e':
		print('[e] executing the population[%d]' % args.population_no)

		if not dfn:
			print(dfn_file_path, 'does not exists')
			return

		if args.input_data == None:
			print('[e] No input_data. Use -input_data')
			return

		input_data = np.load(args.input_data)
		print('input_data', input_data.shape)

		dfn_output_data = dfn.execute_individual_tf(dfn.population[args.population_no],
			input_data)

		if args.output_data:
			output_data = np.load(args.output_data)
			print('output_data', output_data.shape)
			ground_truth = np.argmax(output_data, axis=1)
			print(ground_truth.shape)
			dfn_output = np.argmax(dfn_output_data, axis=1)
			print(dfn_output.shape)
			accuracy = np.mean(ground_truth == dfn_output)
			print('test accuracy', accuracy)

		return

	elif args.mode == 'f':
		print('[f] freezing the population[%d]' % args.population_no)

		if not dfn:
			print(dfn_file_path, 'does not exists')
			return

		frozen_file_path = os.path.join(dfn_dir, args.dfn_name) + '_' + \
			str(args.population_no) + '.pb'
		dfn.freeze_individual_tf(dfn.population[args.population_no],
			frozen_file_path)
		print('population[%d] frozen to %s' % (args.population_no, frozen_file_path))

		return

	if not os.path.exists(dfn_dir):
		os.makedirs(dfn_dir)
	pickle.dump(dfn, open(dfn_file_path, 'wb'))

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('-dnn_name', type=str, help='dnn_name', default=None)
	parser.add_argument('-dfn_name', type=str, help='dfn_name', default=None)
	parser.add_argument('-input_size', type=int, help='input_size', default=-1)
	parser.add_argument('-output_size', type=int, help='output_size', default=-1)
	parser.add_argument('-mode', type=str,	help='mode', default=None)
	parser.add_argument('-population_size', type=int, help='population_size', default=32)
	parser.add_argument('-placeholder_size', type=int, help='placeholder_size', default=16)
	parser.add_argument('-generation', type=int, help='generation', default=100000)
	parser.add_argument('-lamb', type=int, help='lamb', default=4)
	parser.add_argument('-saving_cycle', type=int, help='saving_cycle', default=1)
	parser.add_argument('-input_data', type=str, help='input_data', default=None)
	parser.add_argument('-output_data', type=str, help='output_data', default=None)
	parser.add_argument('-val_input_data', type=str, help='val_input_data', default=None)
	parser.add_argument('-val_output_data', type=str, help='val_output_data', default=None)
	parser.add_argument('-use_distribution', type=int, help='use_distribution', default=1)
	parser.add_argument('-use_neuron_function', type=int, help='use_neuron_function', default=1)
	parser.add_argument('-update_iteration', type=int, help='update_iteration', default=10000)
	parser.add_argument('-update_batch_size', type=int, help='update_batch_size', default=100)
	parser.add_argument('-fitness', type=str, help='fitness', default='mse')
	parser.add_argument('-loss', type=str, help='loss', default='cross_entropy')
	parser.add_argument('-save', type=int, help='save', default=0)
	parser.add_argument('-seed', type=int, help='seed', default=0)
	parser.add_argument('-population_no', type=int,	help='population_no', default=0)

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
