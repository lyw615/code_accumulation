import json, os, sys
import numpy as np
from collections import Counter

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..")))
from code_aculat.visualize.plot_tools import plot_points
import pandas as pd
from matplotlib import pyplot as plt


def get_image2anno(json_path):
    """
    从json里解析出{img_index_in_images：[anno1,anno2]，img2：[anno1,anno2],...]
    可以直接从images字段里解析出对应图片的信息
    Returns:

    """

    with open(json_path, 'r') as f:
        jf = json.load(f)

        imid2img_index = {}
        img_index2annos = {}
        for index, img in enumerate(jf['images']):
            imid2img_index[img['id']] = index

        for anno in jf['annotations']:
            img_index = imid2img_index[anno['image_id']]
            if img_index not in img_index2annos:
                img_index2annos[img_index] = []
            img_index2annos[img_index].append(anno)

        return img_index2annos


def analyse_obs_size_after_resized(json_path, long_short_edges):
    "从coco数据集中获取原图被resize到固定尺寸后，对象的宽高"

    img_index2annos = get_image2anno(json_path)

    with open(json_path, 'r') as f:
        jf = json.load(f)
        images = jf['images']

        for edge in long_short_edges:
            longer, shorter = edge
            resized_wh = []

            for key, value in img_index2annos.items():
                height = images[key]['height']
                width = images[key]['width']

                if width >= height:
                    scale = shorter / height
                    if int(scale * width) > longer:
                        scale = longer / width

                else:
                    scale = shorter / width
                    if int(scale * height) > longer:
                        scale = longer / height

                for anno in value:
                    xmin, ymin, w, h = anno['bbox']
                    x = int(scale * w)
                    y = int(scale * h)
                    resized_wh.append([x, y])

            wh_scale = np.array(resized_wh)
            plot_points([(wh_scale[:, 0], wh_scale[:, 1])], label=shorter)

def analyse_obs_size(json_path):
    "从coco数据集中获取对象的宽高"

    with open(json_path, 'r') as f:
        jf = json.load(f)
    ob_h_w=[]
    for anno in jf['annotations']:

        xmin, ymin, w, h = anno['bbox']

        ob_h_w.append([h, w])

    ob_h_w = np.array(ob_h_w)
    plot_points([(ob_h_w[:, 0], ob_h_w[:, 1])], label='ob_h_w')

def analyse_obs_ratio(json_path):
    "统计对象的ratio,长宽比"

    with open(json_path, 'r') as f:
        jf = json.load(f)
        h_w = []
        for anno in jf['annotations']:
            xmin, ymin, w, h = anno['bbox']

            h_w.append([h, w])
    h_w = np.array(h_w, dtype=np.int)
    ratio = np.ceil(h_w[:, 0] / h_w[:, 1])
    ratio_dict = dict(Counter(ratio))  # 计算给ratio出现的频次

    values = np.array(list(ratio_dict.values()))
    keys = np.array(list(ratio_dict.keys()))
    value_big = np.argsort(values)[::-1]

    number_limit = 15
    if len(value_big) > number_limit:
        value_big = value_big[:number_limit]  # 只统计频数前15

        keys = keys[value_big]
        values = values[value_big]

    keys_ind = np.argsort(keys)  # 按ratio的值排序
    keys = keys[keys_ind]
    values = values[keys_ind]

    "宽高比的值为x轴，对应的数量为y轴"
    array_x = keys
    array_y = values

    hw_dataframe = pd.DataFrame(array_y, columns=["h2w_ratio"])  # 实际还是调的matplotlib可视化
    ax = hw_dataframe.plot(kind='bar', color="#55aacc")
    ax.set_xticklabels(array_x, rotation=0)
    plt.show()

def analyse_image_hw(json_path):
    "统计image长宽比"

    with open(json_path, 'r') as f:
        jf = json.load(f)
        h_w = []
        for img in jf['images']:
            w, h = img['width'],img['height']

            h_w.append([h, w])
    h_w = np.array(h_w)
    plot_points([(h_w[:, 0],h_w[:, 1])], label='image_h_w')

def check_annos(json_path):
    """
    测试annotations里面bbox和segmentation字段为空或者错误的
    Returns:

    """
    with open(json_path, 'r') as f:
        jf = json.load(f)

        for anno in jf["annotations"]:
            seg = anno['segmentation']
            bbox = anno['bbox']

            if len(seg) != 1:  # 不能为空，目标检测时除外
                print(anno)
                raise ("segmentation filed error")

            if len(bbox) != 4:
                print(anno)
                raise ("bbox filed length error")

            xmin, ymin, w, h = bbox
            if xmin < 0 or ymin < 0:  # 左上角坐标为负
                print(anno)
                raise ("bbox filed value error")

            if w < 1 or h < 1:  # 宽高要大于1
                print(anno)
                raise ("bbox filed value error")

def analyse_num_each_class(json_path):
    jf=json.load(open(json_path,'r'))

    obs_dict={}
    for ann in jf['annotations']:
        if ann['category_id'] not in obs_dict.keys():
            obs_dict[ann['category_id']] = 0
        obs_dict[ann['category_id']] += 1

    "得到不重复的类别名称"
    unique_name = list(obs_dict.keys())

    "得到每类对象的数量"
    unique_count = list(obs_dict.values())

    "类别名称为x轴，对应的数量为y轴"
    array_x = unique_name
    array_y = unique_count

    wh_dataframe = pd.DataFrame(array_y, columns=["obs num"])  # 实际还是调的matplotlib可视化
    ax = wh_dataframe.plot(kind='bar', color="#55aacc")
    ax.set_xticklabels(array_x, rotation=0)
    plt.show()