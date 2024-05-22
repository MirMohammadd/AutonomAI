from __future__ import division
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes

def hu_to_grayscale(volume):
    volume = np.clip(volume,-512,512)
    maxVal = np.max(volume)
    minVal = np.min(volume)
    im_volume = (volume - minVal) / max(maxVal - minVal ,1e-3)
    return im_volume * 255

def get_mask_alung(volume):
    volume_im = np.where(volume > 1,1,0)
    shp = volume.shape if hasattr(volume,"shape") else volume
    around_lung = np.zeros((shp[0], shp[1], shp[2]), dtype=np.float32)
    for idx in range(shp[0]):
            around_lung[idx, :, :] = binary_erosion(volume_im[idx], structure=np.ones((15,15))).astype(volume_im.dtype)
    return around_lung

