import os, sys

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from code_aculat.visualize.plot_tools import plot_points


def get_xmls_name_from_csv(csv_file_path):
    "从csv的记录里得到xml的名称"
    xml_list = []

    with open(csv_file_path, "r") as f:
        for line_record in f.readlines():
            xml_name = line_record.strip("\n").split(',')[1] + ".xml"
            xml_list.append(xml_name)
        xml_list.pop(0)
    return xml_list


def visual_image_wh(xml_dir, xmls=None, plot_type='points'):
    """
    本身会传入从csv处相关的xml名称信息,没有则 os.listdir
    :param xml_dir:
    :param xmls:
    :return:
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ÉèÖÃ×ÖÌåÎªºÚÌå
    plt.rcParams['font.family'] = 'sans-serif'  # ÉèÖÃ×ÖÌåÑùÊ½
    plt.rcParams['figure.figsize'] = (10.0, 10.0)  # ÉèÖÃ×ÖÌå´óÐ¡

    if not xmls:
        xmls = os.listdir(xml_dir)

    height_width = []
    hxw = []

    for xml in tqdm(xmls):
        xml_path = os.path.join(xml_dir, xml)

        fp = open(xml_path)

        for p in fp:

            if '<size>' in p:
                size = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(2)]

                height_width.append([size[1], size[0]])
                hxw.append("%s*%s" % (size[1], size[0]))

        fp.close()

    height_width = np.array(height_width)
    if plot_type == "histogram":
        for key in [list(height_width[:, 1]), list(height_width[:, 0]), hxw]:
            "分别展示三张图：宽、高、宽*高"
            unique_values = list(set(key))
            unique_count = [key.count(i) for i in unique_values]

            value2count = np.array([unique_values, unique_count])
            value2count = value2count.transpose([1, 0])

            index_big2low = np.argsort(value2count[:, 1])[::-1]  # 获取从小到大排序的索引，并将索引反序

            number_limit = 15
            if len(index_big2low) > number_limit:
                index_big2low = index_big2low[:number_limit]  # 显示太多容易放不下

            saved_value2count = value2count[index_big2low]

            saved_value2count = saved_value2count[np.argsort(saved_value2count[:, 0])]  # 显示的时候x坐标升序

            wh_dataframe = pd.DataFrame(np.array(saved_value2count[:, 1], dtype=np.uint8),
                                        index=saved_value2count[:, 0], columns=["w,h,w*h"])
            wh_dataframe.plot(kind='bar', color="#55aacc")
            plt.show()

    elif plot_type == "points":  # 宽为x的散点图
        plot_points([(height_width[:, 1], height_width[:, 0])], 'x2width')


def analyse_image_wh(xml_dir, csv_path, plot_type):
    "csv_path 对应多个文件时，说明这些csv文件里记录的xml文件名都要纳入分析范围"
    xml_list = []
    if len(csv_path) > 1:
        for csv in csv_path:
            _xml_list = get_xmls_name_from_csv(csv)
            xml_list.extend(_xml_list)
    else:
        xml_list = os.listdir(xml_dir)

    if len(xml_list) > 0:
        visual_image_wh(xml_dir, xml_list, plot_type)


def output_big_wh(xml_dir, size_thresh, xmls=None):
    """
    根据xml文件里的宽高，输出宽高大于阈值的xml文件
    本身会传入从csv处相关的xml名称信息,没有则 os.listdir
    :param xml_dir:
    :param xmls:
    :return:
    """

    if not xmls:
        xmls = os.listdir(xml_dir)

    big_wh = []

    for xml in tqdm(xmls):
        xml_path = os.path.join(xml_dir, xml)

        fp = open(xml_path)

        for p in fp:

            if '<size>' in p:
                size = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(2)]
                width, height = size

                if width > size_thresh or height > size_thresh:
                    big_wh.append(os.path.basename(xml_path))
    print("这些图片的宽高超出了限制，总数为%d \n" % len(big_wh), big_wh)
    return big_wh


def analyse_obs_per_image(names_resource, xml_dir=None):
    """
    分析数据集中图片里每类对象的数量情况，如果提供的names_resource是csv文件，
    则同时需要提供xml文件夹路径，如果提供的是xml文件夹路径，那么赋给第一个参数就行
    """

    if os.path.isfile(names_resource):
        "如果把xml文件名存在txt或者csv文件里"
        with open(names_resource, 'r', encoding="utf-8") as f:
            names = f.readlines()
    elif os.path.isdir(names_resource):
        "如果提供xml文件夹路径"
        names = os.listdir(names_resource)
        xml_dir = names_resource

    obs_dict = {}
    for _ in tqdm(names):

        xml_path = os.path.join(xml_dir, _)

        fp = open(xml_path)

        for p in fp:
            "这里object后面是类别的名称"
            if '<object>' in p:
                ob_name = next(fp).split('>')[1].split('<')[0]

                if ob_name not in obs_dict.keys():
                    obs_dict[ob_name] = 0
                obs_dict[ob_name] = obs_dict[ob_name] + 1
        fp.close()

    assert len(obs_dict) > 0, "xml文件里没有找到对象信息"
    "得到不重复的类别名称，并排序"
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


def analyse_obs_w2h_ratio(names_resource, xml_dir=None):
    """
       分析数据集中所有对象的宽高比，如果提供的names_resource是csv文件，
       则同时需要提供xml文件夹路径，如果提供的是xml文件夹路径，那么赋给第一个参数就行
    """
    from collections import Counter
    if os.path.isfile(names_resource):
        "如果把xml文件名存在txt或者csv文件里"
        with open(names_resource, 'r', encoding="utf-8") as f:
            names = f.readlines()
    elif os.path.isdir(names_resource):
        "如果提供xml文件夹路径"
        names = os.listdir(names_resource)
        xml_dir = names_resource

    rectangle_position = []
    for _ in tqdm(names):

        xml_path = os.path.join(xml_dir, _)
        fp = open(xml_path)

        for p in fp:
            # if '<size>' in p:
            #     width,height = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(2)]
            if '<bndbox>' in p:
                rectangle = []
                [rectangle.append(round(eval(next(fp).split('>')[1].split('<')[0]))) for _ in range(4)]
                rectangle_position.append(rectangle)

        fp.close()

    assert len(rectangle_position) > 0, "xml文件里没有找到对象信息"

    # 计算所有的宽高比
    rectangle_position = np.array(rectangle_position)
    rec_width = rectangle_position[:, 2] - rectangle_position[:, 0]
    rec_height = rectangle_position[:, 3] - rectangle_position[:, 1]
    w2h_ratio = rec_width / rec_height

    w2h_ratio = w2h_ratio.astype(np.int)
    w2h_ratio = dict(Counter(w2h_ratio))  # 计算给ratio出现的频次

    values = np.array(list(w2h_ratio.values()))
    keys = np.array(list(w2h_ratio.keys()))
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

    wh_dataframe = pd.DataFrame(array_y, columns=["w2h_ratio"])  # 实际还是调的matplotlib可视化
    ax = wh_dataframe.plot(kind='bar', color="#55aacc")
    ax.set_xticklabels(array_x, rotation=0)
    plt.show()


def analyse_obs_scale(names_resource, xml_dir=None):
    """
       分析数据集中每类对象的宽高相对于其所在图片的scale，如果提供的names_resource是csv文件，
       则同时需要提供xml文件夹路径，如果提供的是xml文件夹路径，那么赋给第一个参数就行
    """
    from collections import Counter
    if os.path.isfile(names_resource):
        "如果把xml文件名存在txt或者csv文件里"
        with open(names_resource, 'r', encoding="utf-8") as f:
            names = f.readlines()
    elif os.path.isdir(names_resource):
        "如果提供xml文件夹路径"
        names = os.listdir(names_resource)
        xml_dir = names_resource

    wh_scale = []
    for _ in tqdm(names):

        xml_path = os.path.join(xml_dir, _)
        fp = open(xml_path)

        for p in fp:
            if '<size>' in p:
                width, height = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(2)]
            if '<bndbox>' in p:
                bbox = [(round(eval(next(fp).split('>')[1].split('<')[0]))) for _ in range(4)]
                x = bbox[2] - bbox[0]
                y = bbox[3] - bbox[1]
                wh_scale.append([width // x, height // y])

        fp.close()

    assert len(wh_scale) > 0, "xml文件里没有找到对象信息"
    wh_scale = np.array(wh_scale)
    plot_points([(wh_scale[:, 0], wh_scale[:, 1])], label="w//x with h//y")


def analyse_obs_wh(names_resource, xml_dir=None):
    """
       分析数据集中每类对象的宽高，如果提供的names_resource是csv文件，
       则同时需要提供xml文件夹路径，如果提供的是xml文件夹路径，那么赋给第一个参数就行
    """

    if os.path.isfile(names_resource):
        "如果把xml文件名存在txt或者csv文件里"
        with open(names_resource, 'r', encoding="utf-8") as f:
            names = f.readlines()
    elif os.path.isdir(names_resource):
        "如果提供xml文件夹路径"
        names = os.listdir(names_resource)
        xml_dir = names_resource

    rectangle_position = []
    rect_dict = {}
    for _ in tqdm(names):

        xml_path = os.path.join(xml_dir, _)
        fp = open(xml_path)

        for p in fp:
            if '<object>' in p:
                bnd_box = next(fp).split('>')[1].split('<')[0]
                ob_name = next(fp).split('>')[1].split('<')[0]
                if ob_name not in rect_dict.keys():
                    rect_dict[ob_name] = []

            if '<bndbox>' in p:
                rectangle = []
                [rectangle.append(round(eval(next(fp).split('>')[1].split('<')[0]))) for _ in range(4)]
                if ob_name not in rect_dict.keys():
                    rect_dict[ob_name] = []

                rect_dict[ob_name].append(rectangle)

        fp.close()

    for key, value in rect_dict.items():
        rectangle_position = np.array(value)
        rec_width = rectangle_position[:, 2] - rectangle_position[:, 0]
        rec_height = rectangle_position[:, 3] - rectangle_position[:, 1]

        plot_points([(rec_width, rec_height)], label="%s_width_height" % key)
