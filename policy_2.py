import logging
import numpy as np
import os
import time

from autolab_core import RigidTransform, YamlConfig
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage

from gqcnn import CrossEntropyAntipodalGraspingPolicy, RgbdImageState, ImageGraspSamplerFactory
from gqcnn import Visualizer as vis


#depth_im_filename = os.path.join('data/rgbd/multiple_objects/depth_0.npy')
depth_im_filename = os.path.join('/home/wanfang/ros_depth_image.npy')
camera_intr_filename = 'data/calib/primesense_overhead/primesense_overhead_ros.intr'


# depth_im_filename = os.path.join('2_IMG_DepthMap.npy')
# camera_intr_filename = 'primesense_overhead_photoneo.intr'

config_filename = os.path.join('cfg/examples/policy.yaml')

# read config
config = YamlConfig(config_filename)
inpaint_rescale_factor = config['inpaint_rescale_factor']
policy_config = config['policy']

config['policy']['gqcnn_model'] = 'GQ-Image-Wise'

# setup sensor
camera_intr = CameraIntrinsics.load(camera_intr_filename)

# read images
depth_im = DepthImage.open(depth_im_filename, frame=camera_intr.frame)
depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
                      frame=camera_intr.frame)

# create state
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
state = RgbdImageState(rgbd_im, camera_intr)

# init policy
policy = CrossEntropyAntipodalGraspingPolicy(policy_config)
policy_start = time.time()
action = policy(state)
logging.info('Planning took %.3f sec' %(time.time() - policy_start))

# vis final grasp
if policy_config['vis']['final_grasp']:
    vis.figure(size=(10,10))
    vis.imshow(rgbd_im.depth, vmin=0, vmax=1.3)
    vis.grasp(action.grasp, scale=2.5, show_center=True, show_axis=True)
    vis.title('Planned grasp on depth (Q=%.3f)' %(action.q_value))
    vis.show()

grasp_sampler = ImageGraspSamplerFactory.sampler(policy_config['sampling']['type'], policy_config['sampling'], policy_config['gripper_width'])
grasp_sampler.sample(rgbd_im, camera_intr,
                                    policy_config['num_seed_samples'],
                                    visualize=policy_config['vis']['grasp_sampling'],
                                    seed=None)
