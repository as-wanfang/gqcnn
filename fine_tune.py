import argparse
import logging
import time
import os

from autolab_core import YamlConfig
from gqcnn import GQCNN, SGDOptimizer, GQCNNAnalyzer
import autolab_core.utils as utils

if __name__ == '__main__':

	#model_dir: /home/yangyi/dex-net2.0-custom/src/gqcnn/GQ-training/model
	config_filename = os.path.join('/home/yangyi/dex-net2.0-custom/src/gqcnn/cfg/tools/training.yaml')

	train_config = YamlConfig(config_filename)
	gqcnn_config = train_config['gqcnn_config']

	def get_elapsed_time(time_in_seconds):
		""" Helper function to get elapsed time """
		if time_in_seconds < 60:
			return '%.1f seconds' % (time_in_seconds)
		elif time_in_seconds < 3600:
			return '%.1f minutes' % (time_in_seconds / 60)
		else:
			return '%.1f hours' % (time_in_seconds / 3600)

	start_time = time.time()
	model_dir = train_config['model_dir']
	gqcnn = GQCNN.load(model_dir)
	sgdOptimizer = SGDOptimizer(gqcnn, train_config)
	with gqcnn._graph.as_default():
	        sgdOptimizer.optimize()
	logging.info('Total Fine Tuning Time:' + str(get_elapsed_time(time.time() - start_time)))
