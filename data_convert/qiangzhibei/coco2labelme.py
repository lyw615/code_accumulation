# -*- coding: utf-8 -*-
"""
@author: Taoting
将用coco格式的json转化成labeime标注格式的json
"""

import json
import cv2
import numpy as np
import os
import pycocotools.mask as mask


def polygonFromMask(maskedArr):  # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py

    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
        else:
            continue

    # 不用再转成RLE了
    #     RLEs = mask.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    #
    # RLE = mask.merge(RLEs)
    # # RLE = mask.encode(np.asfortranarray(maskedArr))
    # area = mask.area(RLE)

    [x, y, w, h] = cv2.boundingRect(maskedArr)  # 这个用于生成外接矩形

    return segmentation  # , [x, y, w, h], area


# 用一个labelme格式的json作为参考，因为很多信息都是相同的，不需要修改。
def reference_labelme_json(ref_json_path):
    data = json.load(open(ref_json_path))
    return data


def labelme_shapes(data_ref, anno_list, id2category, score_thresh):
    shapes = []
    # label_num = {'person': 0, 'bicycle': 0, 'car': 0, 'motorcycle': 0, 'bus': 0, 'train': 0, 'truck': 0}  # 根据你的数据来修改
    label_num = {'ship': 0}  # 根据你的数据来修改
    for ann in anno_list:
        shape = {}

        if ann['score'] < score_thresh:  # 分数过滤
            continue
        class_name = id2category[ann['category_id']]
        # label要对应每一类从_1开始编号
        # label_num[class_name[0]] += 1
        shape['label'] = class_name
        shape['line_color'] = data_ref['shapes'][0]['line_color']
        shape['fill_color'] = data_ref['shapes'][0]['fill_color']

        shape['points'] = []
        # ~ print(ann['segmentation'])
        if isinstance(ann['segmentation'], dict):
            maskedArr = mask.decode(ann['segmentation'])
            # from matplotlib import pyplot as plt
            # plt.imshow(maskedArr)
            # plt.show()
            segmentation = polygonFromMask(maskedArr)

            if len(segmentation) > 0:

                x = segmentation[0][::2]  # 奇数个是x的坐标
                y = segmentation[0][1::2]  # 偶数个是y的坐标
            else:
                continue
        elif isinstance(ann['segmentation'], list):
            x = ann['segmentation'][0][::2]  # 奇数个是x的坐标
            y = ann['segmentation'][0][1::2]  # 偶数个是y的坐标

        else:
            continue
        for j in range(len(x)):
            shape['points'].append([x[j], y[j]])

        shape['shape_type'] = data_ref['shapes'][0]['shape_type']
        shape['flags'] = data_ref['shapes'][0]['flags']
        shapes.append(shape)

    return shapes


def Coco2labelme(data_ref, id2annos, id2index, images, id2category, out_dir, score_thresh, corre_path):
    for image_id in id2annos.keys():

        data_labelme = {}
        # data_labelme['version'] = data_ref['version']
        data_labelme['version'] = "4.5.9"
        data_labelme['flags'] = data_ref['flags']

        data_labelme['shapes'] = labelme_shapes(data_ref, id2annos[image_id], id2category, score_thresh)

        if len(data_labelme['shapes']) < 1:
            continue  # 说明没有对象

        data_labelme['lineColor'] = data_ref['lineColor']
        data_labelme['fillColor'] = data_ref['fillColor']

        # 写入json文件和对应Image文件夹的相对路径
        data_labelme['imagePath'] = "%s/%s" % (corre_path, images[id2index[image_id]]['file_name'])

        data_labelme['imageData'] = None
        # data_labelme['imageData'] = data_ref['imageData']

        data_labelme['imageHeight'] = images[id2index[image_id]]['height']
        data_labelme['imageWidth'] = images[id2index[image_id]]['width']

        file_name = images[id2index[image_id]]['file_name'].replace('.bmp', '.json')
        # 保存json文件
        json.dump(data_labelme, open(os.path.join(out_dir, file_name), 'w'), indent=4)


def sparse_mmdet_out_json(test_json, pre_json):
    json_t = json.load(open(test_json, 'r'))
    json_p = json.load(open(pre_json, 'r'))

    id2annos = {}

    for pre in json_p:
        if pre['image_id'] not in id2annos:
            id2annos[pre['image_id']] = []
        id2annos[pre['image_id']].append(pre)

    id2index = {}
    for _ in range(len(json_t['images'])):
        image = json_t['images'][_]
        id2index[image['id']] = _

    images = json_t['images']
    category = json_t['categories']
    id2category = {}
    for cate in category:
        if cate['id'] not in id2category:
            id2category[cate['id']] = cate['name']
    return id2annos, id2index, images, id2category


if __name__ == '__main__':
    out_dir = r'G:\hrsc\mask'
    os.makedirs(out_dir, exist_ok=True)

    test_json = r"G:\target_img.json"  # mmdet的test json 和预测出的json
    pre_json = r"G:\ship_mask_1000.segm.json"
    ref_json_path = r'G:\mask\ship_train_7.json'  # labelme创建的json文件，相关信息好参照

    score_thresh = 0.5  # 过滤掉分数低的结果
    corre_path = "..\Images"  # label json 与原图的相对位置

    # 参考的json
    data_ref = reference_labelme_json(ref_json_path)

    id2annos, id2index, images, id2category = sparse_mmdet_out_json(test_json, pre_json)

    data_labelme = Coco2labelme(data_ref, id2annos, id2index, images, id2category, out_dir, score_thresh, corre_path)
