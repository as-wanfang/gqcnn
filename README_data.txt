# DEX-NET 3.0 DATASET

# OVERVIEW
This is the latest version of the dataset used to train the Grasp-Quality-Convolutional-Neural-Network from Dex-Net 3.0.
The dataset contains approximately 2.8 million datapoints generated from 1,500 3D object models from the KIT and 3DNet datasets.
The dataset was generated on September 13, 2017.
For more information on the dataset generation method, please see our paper at berkeleyautomation.github.io/dex-net.

If you use this dataset in a publication, please cite:

Jeffrey Mahler, Matthew Matl, Xinyu Liu, Albert Li, David Gealy, and Ken Goldberg. "Dex-Net 3.0: Computing Robust Robot Suction Grasp Targets using a New Analytic Model and Deep Learning." IEEE International Conference on Robotics and Automation, 2018 (Under Review). Brisbane, Australia.

# DATA EXTRACTION
unzip dexnet_3.zip

# DATA FORMAT
Files are in compressed numpy (.npz) format and organized by attribute.
Each file contains 1,000 datapoints except for the last file (6728), which contains 850 datapoints.
There are five different attributes:

  1) depth_ims_tf_table:
       Description: depth images transformed to align the grasp center with the image center and the grasp axis with the middle row of pixels
       File dimension: 1000x32x32x1 (except the last file)
       Organization: {num_datapoints} x {image_height} x {image_width} x {num_channels}
       Notes: Rendered with OSMesa using the parameters of a Primesense Carmine 1.08

  2) hand_poses:
       Description: configuration of the robot gripper corresponding to the grasp
       File dimension: 1000x7 (except the last file)
       Organization: {num_datapoints} x {hand_configuration}, where columns are
         0: row index, in pixels, of grasp center projected into a depth image centered on the object (pre-rotation-and-translation)
         1: column index, in pixels, of grasp center projected into a depth image centered on the object (pre-rotation-and-translation)
         2: depth, in meters, of gripper center from the camera that took the corresponding depth image
	 3: 
	 4: angle, in radians, of the grasp axis from the image x-axis (middle row of pixels, pointing right in image space)
	 5: row index, in pixels, of the object center projected into a depth image centered on the world origin
	 6: column index, in pixels, of the object center projected into a depth image centered on the world origin
       Notes: To replicate the Dex-Net 3.0 results, you only need columns 2 and 3.
         The suction cup diameter was 15mm.

  3) robust_suction_wrench_resistance:
       Description: value of robust wrench resistance under gravity using the Dex-Net 3.0 compliant suction contact model 
       File dimension: 1000 (except the last file)
       Organization: {num_datapoints}
       Notes: Threshold the values in this value by 0.2 to generate the 0-1 labels of Dex-Net 2.0

# DATA INDEXING
The index into the first dimension of each file corresponds to a different attribute.
Furthermore, the same index in the file with the same number corresponds to the same datapoints.

For example, datapoint 2478 could be accessed in Python as:
  depth_im = np.load('depth_ims_tf_table_00002.npz')['arr_0'][478,...]
  hand_pose = np.load(hand_poses_00002.npz')['arr_0'][478,...]
  grasp_metric = np.load(robust_suction_wrench_resistance_00002.npz')['arr_0'][478,...]
