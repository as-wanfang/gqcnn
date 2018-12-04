import numpy as np
from PIL import Image

im = Image.open('3_IMG_DepthMap.tif')

d = np.array(im)
#np.save('2_IMG_DepthMap_downsampled.npy', d[::2,::2]/1000)
np.save('3_IMG_DepthMap.npy', d[::2,::2]/1000)
