import os
import csv
import cv2, rasterio
import json
import shutil
from tqdm import tqdm
import numpy as np


def json_from_imagedir():
    # {"echinus": 1,"scallop": 2,"starfish": 3,"holothurian": 4}
    train_img_dir = '/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/test/TIFFImages'
    train_img_dir = '/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/train/TIFFImages'
    annot_dst_path = '/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/train/pre.json'
    js = {'images': [], 'annotations': [],
          'categories': [{'id': 1, 'name': "Airport"}, {'id': 2, 'name': "Port"}], }  # {'id': 5, 'name': "waterweeds"}
    # {"echinus": 1, "scallop": 2, "starfish": 3, "holothurian": 4}
    annot_id = 0
    img_id = 0
    img_names = os.listdir(train_img_dir)
    for img_name in tqdm(img_names):
        img_path = os.path.join(train_img_dir, img_name)

        if img_name.endswith('tif'):
            with  rasterio.open(img_path) as ds:
                height = ds.height
                width = ds.width
        else:
            # img = cv2.imread(img_path)
            # height = img.shape[0]
            # width = img.shape[1]
            continue

        js['images'].append({'file_name': img_name, 'height': height, 'width': width, 'id': img_id})
        js['annotations'].append(
            {'segmentation': [], 'area': 1, 'iscrowd': 0, 'image_id': img_id, 'bbox': [1, 1, 1, 1],
             'category_id': 0, 'id': annot_id})

        annot_id += 1
        img_id += 1

    with open(annot_dst_path, 'w') as f:
        json.dump(js, f)


if __name__ == '__main__':
    "mmdetection need json list  as image list to be predicted"
    json_from_imagedir()
