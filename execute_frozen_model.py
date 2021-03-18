import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import sys
import argparse
import os


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    return graph

def main(args):

	if args.frozen_model_filename is None:
		print('no frozen_model_filename')
		return

	if args.input_data is None:
		print('no input_data')
		return

	trace_filename = args.trace_filename
	if trace_filename is None:
		trace_filename = os.path.basename(args.frozen_model_filename) + '.json'

	input_data = np.load(args.input_data)
	input_feed_data = np.reshape(input_data[:args.input_feed_size], [-1, 45*80])
	print(input_feed_data.shape)

	graph = load_graph(args.frozen_model_filename)

	#for op in graph.get_operations():
	#	print(op.name)

	x = graph.get_tensor_by_name('import/output-1:0')
	y = graph.get_tensor_by_name('import/output:0')

	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()

	with tf.Session(graph=graph) as sess:
		y_out = sess.run(y, feed_dict={x:input_feed_data},
			options=run_options, run_metadata=run_metadata)
		tl = timeline.Timeline(run_metadata.step_stats)
		ctf = tl.generate_chrome_trace_format()
		with open(trace_filename, 'w') as f:
			f.write(ctf)
	#print(y_out)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('-input_data', type=str, help='input_data', default=None)
	parser.add_argument('-frozen_model_filename', type=str, help='frozen_model_filename', default=None)
	parser.add_argument('-trace_filename', type=str, help='trace_filename', default=None)
	parser.add_argument('-input_feed_size', type=int, help='input_feed_size', default=100)
	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
