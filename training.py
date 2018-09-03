# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Script for training a Grasp Quality Neural Network (GQ-CNN) from scratch.

Author
------
Vishal Satish & Jeff Mahler
"""
import argparse
import logging
import os
import time
import os

import autolab_core.utils as utils
from autolab_core import YamlConfig
from gqcnn import GQCNN, GQCNNOptimizer, GQCNNAnalyzer
from gqcnn import utils as gqcnn_utils


    # parse args
    # parser = argparse.ArgumentParser(description='Train a Grasp Quality Convolutional Neural Network from scratch with TensorFlow')
    # parser.add_argument('dataset_dir', type=str, default=None,
    #                     help='path to the dataset to use for training and validation')
    # parser.add_argument('--split_name', type=str, default=None,
    #                     help='name of the split to train on')
    # parser.add_argument('--output_dir', type=str, default=None,
    #                     help='path to store the model')
    # parser.add_argument('--tensorboard_port', type=int, default=6006,
    #                     help='port to launch tensorboard on')
    # parser.add_argument('--seed', type=int, default=None,
    #                     help='random seed for training')
    # parser.add_argument('--config_filename', type=str, default=None,
    #                     help='path to the configuration file to use')
    # parser.add_argument('--unique_name', type=bool, default=False,
    #                     help='add a unique name to dataset path which encodes the current date')
dataset_dir = 'dexnet_09_13_17'
split_name = 'image_wise'

# set default config filename
# config_filename = 'cfg/tools/train_suction.yaml'
config_filename = 'cfg/training.yaml'

# open train config
train_config = YamlConfig(config_filename)
gqcnn_params = train_config['gqcnn']

# create a unique output folder based on the date and time
unique_name = time.strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join('models', unique_name)
utils.mkdir_safe(output_dir)

# train the network
start_time = time.time()
gqcnn = GQCNN(gqcnn_params)
optimizer = GQCNNOptimizer(gqcnn,dataset_dir,split_name,output_dir,train_config)

optimizer.train()
