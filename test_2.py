from gqcnn import GQCNN, DeepOptimizer, GQCNNAnalyzer
from autolab_core import YamlConfig
import time
import logging
import numpy as np
from PIL import Image

# train_config = YamlConfig('cfg/tools/train_grasp_quality_cnn_dexnet_large.yaml')
train_config = YamlConfig('cfg/tools/train_dex-net_2.0.yaml')
gqcnn_config = train_config['gqcnn_config']

# change this path to your own directorys
model_dir = '/home/wanfang/catkin_ws/src/gqcnn/GQ-Image-Wise'

gqcnn = GQCNN.load(model_dir)
img = np.load('/home/wanfang/catkin_ws/src/gqcnn/data/rgbd/bar_clamp/depth_0.npy')
img = img[::3,::3]

im = np.reshape(img,(160,214))
im = Image.fromarray(np.uint8(im * 255) , 'L')
im.save('test.png')
t = im.crop((90,50,122,82))
t.save('crop.png')
crop = img[50:82,90:122]
images = np.reshape(crop,(1,32,32,1))
poses = np.ones([1,1])

gqcnn.predict(images, poses)
