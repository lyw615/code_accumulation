import os, json
import numpy as np


def analy_no_pred_image_size(json_file, image_source):
    """
    分析没有预测结果,图片的size信息
    Args:
        json_file:存有图片size信息的json
        image_source:

    Returns:

    """
    # js=json.load(open("/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold-v1/fold_v2/train.json", "r"))
    # image_source=r"/home/data1/yw/github_projects/personal_github/code_aculat/data_operation/image_pre_none.txt"
    js = json.load(open(json_file, "r"))

    with open(image_source, 'r') as tt:
        records = tt.readlines()

    image_name2info = {}
    for dict in js['images']:
        image_name2info[dict['file_name']] = [dict['height'], dict['width']]

    for tif in records:
        tif_name = tif.strip('\n')

        print('tif name  %s  ,height  is  %d  width  is  %d' % (
        tif_name, image_name2info[tif_name][0], image_name2info[tif_name][1]))
