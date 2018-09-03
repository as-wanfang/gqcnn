import logging
import numpy as np
import os
import time

from autolab_core import RigidTransform, YamlConfig
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from gqcnn.visualizer import Visualizer as vis
from gqcnn import CrossEntropyRobustGraspingPolicy, RgbdImageState

# depth_im_filename = os.path.join('data/rgbd/bar_clamp/depth_0.npy')
# camera_intr_filename = 'data/calib/primesense.intr'


depth_im_filename = os.path.join('3_IMG_DepthMap.npy')
camera_intr_filename = 'primesense_overhead_photoneo.intr'

config_filename = os.path.join('cfg/examples/dex-net_3.0.yaml')

# read config
config = YamlConfig(config_filename)
inpaint_rescale_factor = config['inpaint_rescale_factor']
policy_config = config['policy']

policy_config['metric']['gqcnn_model'] = 'models/GQCNN-3.0'

# setup sensor
camera_intr = CameraIntrinsics.load(camera_intr_filename)

# read images
depth_im = DepthImage.open(depth_im_filename, frame=camera_intr.frame)
depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
                      frame=camera_intr.frame)

# need mask if the area of interest is small relative to the whole depth image
mask = np.ones([depth_im.height, depth_im.width]).astype(np.uint8)
# mask[160:260, 270:370] = 2 %
mask[90:400, 200:670] = 2

# mask = (depth_im.data<0.59).astype(np.uint8)+1
segmask = BinaryImage(mask, frame=camera_intr.frame, threshold=1.5)

# create state
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

# init policy
policy = CrossEntropyRobustGraspingPolicy(policy_config)
policy_start = time.time()
action = policy(state)
logging.info('Planning took %.3f sec' %(time.time() - policy_start))

print('Angle: %.3f' %action.grasp.angle)
print('Approach_angle: %.3f' %action.grasp.approach_angle)
# vis final grasp
if policy_config['vis']['final_grasp']:
    vis.figure(size=(10,10))
    vis.imshow(rgbd_im.depth, vmin=0, vmax=1.3)
    vis.scatter(action.grasp.center.x, action.grasp.center.y)
    vis.title('Planned grasp on depth (Q=%.3f)' %(action.q_value))
    vis.show()
