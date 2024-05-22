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


def get_mask(seg):
    shp = seg.shape
    lung = np.zeros((shp[0], shp[1], shp[2]), dtype=np.float32)
    lung[np.equal(seg,255)] = 255

    return lung
def get_FOV(around_lung, lung):
    FOV = np.where((around_lung + lung) >0, 1, 0)
    for idx in range(FOV.shape[0]):
        FOV[idx, :, :] = binary_fill_holes(FOV[idx, :, :], structure=np.ones((5,5))).astype(FOV.dtype)
    return FOV

def return_axials(vol, seg):

    # Prepare segmentation and volume
    vol = vol.get_data()
    seg = seg.get_data()
    seg = seg.astype(np.int32)
    
    # Convert to a visual format
    vol_ims = hu_to_grayscale(vol)
    lung    = get_mask(seg)
    around_lung = get_mask_alung(vol_ims)
    FOV = get_FOV(around_lung, lung)
    around_lung = np.where((FOV - lung) >0, 1, 0)

    return vol_ims, lung, around_lung, FOV

    
