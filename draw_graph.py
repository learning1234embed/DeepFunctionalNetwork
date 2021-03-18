from __future__ import print_function
import os
import pickle 
import argparse
import sys

from dfn import DFN

def main(args):
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

	if args.mode == 'd':
		print('[d] drawing a deep functional network')
		filename = args.dnn_name + '_' + args.dfn_name + '_' + \
			str(args.population_no) + '.png'
		dfn.population[args.population_no].draw_plain_graph(filename)
		return

	if args.mode == 'i':
		print('[i] drawing a deep functional network')
		filename = args.dnn_name + '_' + args.dfn_name + '_' + \
			str(args.population_no) + '_interpretable.png'
		dfn.population[args.population_no].draw_interpretable_graph(filename)
		return

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('-dnn_name', type=str, help='dnn_name', default=None)
	parser.add_argument('-dfn_name', type=str, help='dfn_name', default=None)
	parser.add_argument('-mode', type=str,	help='mode', default=None)
	parser.add_argument('-population_no', type=int,	help='population_no', default=0)

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
