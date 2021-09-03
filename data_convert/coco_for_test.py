import os
import csv
import cv2, rasterio
import json
import shutil
from tqdm import tqdm
import numpy as np


def json_from_imagedir(images_dir, annot_dst_path, train_img_source=None):
    "create json from image dir or txt/csv file ,set to train_img_source when the latter"

    # {"echinus": 1,"scallop": 2,"starfish": 3,"holothurian": 4}
    js = {'images': [], 'annotations': [],
          # 'categories': [{'id': 1, 'name': "Airport"}, {'id': 2, 'name': "Port"}],
          # 'categories': [{'id': 1, 'name': "ship"}],
          'categories': [{'id': 1, 'name': "23"},{'id': 2, 'name': "22"},{'id': 3, 'name': "21"},{'id': 4, 'name': "2"},{'id': 5, 'name': "1"},{'id': 6, 'name': "15"}],
          }  # {'id': 5, 'name': "waterweeds"}
    # {"echinus": 1, "scallop": 2, "starfish": 3, "holothurian": 4}

    if not train_img_source:
        train_img_source = images_dir

    if os.path.isdir(train_img_source):
        img_names = os.listdir(train_img_source)

    elif os.path.isfile(train_img_source):  # support csv or txt file, which records image names
        with open(train_img_source, 'r') as f:
            img_names = f.readlines()
        img_names = [x.strip('\n') for x in img_names]

    annot_id = 0
    img_id = 0

    for img_name in tqdm(img_names):
        img_path = os.path.join(images_dir, img_name)

        if img_name.endswith('tif'):
            with  rasterio.open(img_path) as ds:
                height = ds.height
                width = ds.width
        else:
            img = cv2.imread(img_path)
            height = img.shape[0]
            width = img.shape[1]
            # continue

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
    train_img_source = '/home/data1/yw/data/compt_data/qzb_ship/target_img.txt'
    train_img_source = '/home/data1/yw/data/iobjectspy_out/mmdetection/show_fine'
    images_dir = '/home/data1/yw/data/compt_data/qzb_ship/k-fold/fold_v1/Images'
    annot_dst_path = '/home/data1/yw/data/compt_data/qzb_ship/valid_show_fine.json'
    json_from_imagedir(images_dir, annot_dst_path, train_img_source)
