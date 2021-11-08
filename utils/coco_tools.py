from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
import numpy as np
import json


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

def draw_ann_masks(anns, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    return annotation's segmentation to masks
    """
    mask_zero = np.zeros(shape=(height, width))
    for ann in anns:
        mask = annToMask(ann, height, width)
        mask_zero += mask
    plt.imshow(mask_zero)
    plt.show()


def annToRLE(ann, height, width):
    """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # RLE
        rle = ann['segmentation']
    return rle


def annToMask(ann, height, width):
    """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to
        binary mask.
        :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m