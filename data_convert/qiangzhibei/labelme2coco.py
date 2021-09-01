# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json, os
import matplotlib.pyplot as plt
import skimage.io as io
import cv2, random
from labelme import utils
import numpy as np
import glob
from tqdm import tqdm
import PIL.Image


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):
    def __init__(self, labelme_json=[], fix_categories2id=None, save_json_path='./tran.json', suffix='.bmp'):
        '''
        :param labelme_json: ËùÓÐlabelmeµÄjsonÎÄ¼þÂ·¾¶×é³ÉµÄÁÐ±í
        :param save_json_path: json±£´æÎ»ÖÃ
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = fix_categories2id
        self.annotations = []
        # self.data_coco = {}
        # self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.suffix = suffix
        self.save_json()

    def data_transfer(self):

        for num, json_file in enumerate(tqdm(self.labelme_json)):
            with open(json_file, 'r') as fp:
                file_name = os.path.basename(json_file).split('.')[0] + self.suffix
                data = json.load(fp)  # ¼ÓÔØjsonÎÄ¼þ
                self.images.append(self.image(data, num, file_name))
                for shapes in data['shapes']:
                    label = shapes['label']

                    # if label == "ship":
                    #     continue

                    # if label not in self.label:
                    #     self.categories.append(self.categorie(label))
                    #     self.label.append(label)
                    # points.append([points[0][0],points[1][1]])
                    # points.append([points[1][0],points[0][1]])

                    points = shapes['points']  # ÕâÀïµÄpointÊÇÓÃrectangle±ê×¢µÃµ½µÄ£¬Ö»ÓÐÁ½¸öµã£¬ÐèÒª×ª³ÉËÄ¸öµã
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num, file_name):
        image = {}
        image['height'] = data['imageHeight']
        image['width'] = data['imageWidth']
        image['id'] = num + 1

        image['file_name'] = file_name
        self.height = data['imageHeight']
        self.width = data['imageWidth']

        return image

    # def categorie(self, label):
    #     categorie = {}
    #     categorie['supercategory'] = 'Cancer'
    #     categorie['id'] = len(self.label) + 1  # 0 Ä¬ÈÏÎª±³¾°
    #     categorie['name'] = label
    #     return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # Ê¹ÓÃlist±£´æjsonÎÄ¼þÊ±±¨´í£¨²»ÖªµÀÎªÊ²Ã´£©
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] Ê¹ÓÃ¸Ã·½Ê½×ª³Élist
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = self.categories[label]  # ×¢Òâ£¬Ô´´úÂëÄ¬ÈÏÎª1
        annotation['id'] = self.annID
        return annotation

    # def getcatid(self, label):
    #     for categorie in self.categories:
    #         if label == categorie['name']:
    #             return categorie['id']
    #     return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # »­±ß½çÏß
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # »­¶à±ßÐÎ ÄÚ²¿ÏñËØÖµÎª1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''´Ómask·´Ëã³öÆä±ß¿ò
        mask£º[h,w]  0¡¢1×é³ÉµÄÍ¼Æ¬
        1¶ÔÓ¦¶ÔÏó£¬Ö»Ðè¼ÆËã1¶ÔÓ¦µÄÐÐÁÐºÅ£¨×óÉÏ½ÇÐÐÁÐºÅ£¬ÓÒÏÂ½ÇÐÐÁÐºÅ£¬¾Í¿ÉÒÔËã³öÆä±ß¿ò£©
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # ½âÎö×óÉÏ½ÇÐÐÁÐºÅ
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # ½âÎöÓÒÏÂ½ÇÐÐÁÐºÅ
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] ¶ÔÓ¦COCOµÄbbox¸ñÊ½

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        categories_list = []

        data_coco['images'] = self.images

        for key, value in self.categories.items():
            categories_list.append({'name': key, 'id': value})

        data_coco['categories'] = categories_list
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # ±£´æjsonÎÄ¼þ
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 ¸ü¼ÓÃÀ¹ÛÏÔÊ¾


def convert(labelme_json_dir, outdir, fix_categories2id, portion=0.8, split_source=None, suffix=None):
    js_train_val = os.listdir(labelme_json_dir)
    random.shuffle(js_train_val)

    if split_source:
        with open(os.path.join(split_source, 'new_train.csv'), 'r') as f:
            lines = f.readlines()
            train_port = [os.path.join(labelme_json_dir, x.strip('\n') + ".json") for x in lines]

        with open(os.path.join(split_source, 'new_test.csv'), 'r') as f:
            lines = f.readlines()
            val_port = [os.path.join(labelme_json_dir, x.strip('\n') + ".json") for x in lines]
    else:
        # split json to train val
        portion = 0.8
        train_num = int(len(js_train_val) * portion)
        train_port = js_train_val[:train_num]
        val_port = js_train_val[train_num:]

        train_port = [os.path.join(labelme_json_dir, x) for x in train_port]
        val_port = [os.path.join(labelme_json_dir, x) for x in val_port]

    # labelme_json=['./Annotations/*.json']

    labelme2coco(train_port, fix_categories2id, os.path.join(outdir, "train.json"), suffix)
    labelme2coco(val_port, fix_categories2id, os.path.join(outdir, "val.json"), suffix)


fix_categories2id = {"1": 1, "2": 2, "15": 3, "21": 4, "22": 5, "23": 6}  # 每次都要硬编码这个类别和id的对应关系，避免训练集和验证集的不一样
fix_categories2id = {"14": 14}  # 每次都要硬编码这个类别和id的对应关系，避免训练集和验证集的不一样
suffix = '.bmp'
labelme_json_dir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\outship_cls_num\big_41\41\mask"
outdir = r"G:\hrsc\out_coco"
# csv_dir = r"E:\k-fold-fine\fold_v1"
os.makedirs(outdir, exist_ok=True)
convert(labelme_json_dir, outdir, fix_categories2id, split_source=None, suffix=suffix)
