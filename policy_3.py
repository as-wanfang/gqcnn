import logging
import numpy as np
import os
import time

from autolab_core import RigidTransform, YamlConfig
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from gqcnn.visualizer import Visualizer as vis
from gqcnn import CrossEntropyRobustGraspingPolicy, RgbdImageState

from PIL import Image
im = Image.open("4_IMG_DepthMap.tif")
imarray = np.array(im)
imarray = imarray/1000
np.save("4_IMG_DepthMap.npy", imarray)
# depth_im_filename = os.path.join('data/rgbd/bar_clamp/depth_0.npy')
# camera_intr_filename = 'data/calib/primesense.intr'


depth_im_filename = os.path.join('4_IMG_DepthMap.npy')
camera_intr_filename = 'photoneo_overhead.intr'

config_filename = os.path.join('cfg/ros_nodes/grasp_planner_node.yaml')

# read config
config = YamlConfig(config_filename)
inpaint_rescale_factor = config['inpaint_rescale_factor']
policy_config = config['policy']

policy_config['metric']['gqcnn_model'] = '/home/bionicdl/Downloads/gqcnn_models/GQCNN-3.0'

# setup sensor
camera_intr = CameraIntrinsics.load(camera_intr_filename)

# read images
depth_im = DepthImage.open(depth_im_filename, frame=camera_intr.frame)
depth_im_b = DepthImage.open('depth_map_background.npy', frame=camera_intr.frame)
depth_im_s = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
                      frame=camera_intr.frame)

# need mask if the area of interest is small relative to the whole depth image
mask = np.ones([depth_im.height, depth_im.width]).astype(np.uint8)
mask[:, 245:900] = 2
mask = mask + ((depth_im.data)<0.5).astype(np.uint8) + ((depth_im.data)>0.0).astype(np.uint8)
segmask = BinaryImage(mask, frame=camera_intr.frame, threshold=3.5)
vis.figure(size=(10,10))
vis.imshow(segmask, vmin=0, vmax=1)
vis.show()

# create state
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im_s)
state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
# point_cloud_im = camera_intr.deproject_to_image(depth_im)
# normal_cloud_im = point_cloud_im.normal_cloud_im()
#
# mask = ((depth_im_b.data - depth_im.data)>5).astype(np.uint8) + ((depth_im_b.data - depth_im.data)<100).astype(np.uint8) + (depth_im.data>100).astype(np.uint8) + (np.sum(normal_cloud_im.data*normal_cloud_im.data,axis=2)>0).astype(np.uint8)
# segmask = BinaryImage(mask, frame=camera_intr.frame, threshold=3.5)
# state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

# init policy
policy = CrossEntropyRobustGraspingPolicy(policy_config)
policy_start = time.time()
action = policy(state)
logging.info('Planning took %.3f sec' %(time.time() - policy_start))

print('Angle: %.3f' %action.grasp.angle)
print('Approach_angle: %.3f' %action.grasp.approach_angle)
# vis final grasp
# if policy_config['vis']['final_grasp']:
vis.figure(size=(10,10))
vis.imshow(depth_im, vmin=0.4, vmax=0.5)
vis.scatter(action.grasp.center.x, action.grasp.center.y)
vis.title('Planned grasp on depth (Q=%.3f)' %(action.q_value))
vis.show()
