import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def get_xmls_name_from_csv(csv_file_path):
    "从csv的记录里得到xml的名称"
    xml_list = []

    with open(csv_file_path, "r") as f:
        for line_record in f.readlines():
            xml_name = line_record.strip("\n").split(',')[1] + ".xml"
            xml_list.append(xml_name)
        xml_list.pop(0)
    return xml_list


def visual_image_wh(xml_dir, xmls=None):
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

    width_list = []
    height_list = []
    width_height = []

    for xml in tqdm(xmls):
        xml_path = os.path.join(xml_dir, xml)

        fp = open(xml_path)

        for p in fp:

            if '<size>' in p:
                rectangle = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(2)]

                width_list.append(rectangle[0])
                height_list.append(rectangle[1])
                width_height.append("%s*%s" % (rectangle[0], rectangle[1]))

        fp.close()

    for key in [width_list, height_list, width_height]:
        "分别展示三张图：宽、高、宽*高"
        unique_ratio = set(key)
        unique_ratio = sorted(unique_ratio)
        bbox_unique_count = [key.count(i) for i in unique_ratio]

        wh_dataframe = pd.DataFrame(bbox_unique_count, index=unique_ratio, columns=["wh"])
        wh_dataframe.plot(kind='bar', color="#55aacc")
        plt.show()


def analyse_image_wh(xml_dir, csv_path):
    "csv_path 对应多个文件时，说明这些csv文件里记录的xml文件名都要纳入分析范围"
    xml_list = []
    if len(csv_path) > 1:
        for csv in csv_path:
            _xml_list = get_xmls_name_from_csv(csv)
            xml_list.extend(_xml_list)
    visual_image_wh(xml_dir, xml_list)


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

    obs_num = []
    for _ in tqdm(names):
        ob_num = 0
        xml_path = os.path.join(xml_dir, _)

        fp = open(xml_path)

        for p in fp:
            "这里object后面是类别的名称"
            if '<object>' in p:
                _name = next(fp).split('>')[1].split('<')[0]
                ob_num += 1
        fp.close()
        obs_num.append(ob_num)

    "得到不重复的类别名称，并排序"
    unique_value = set(obs_num)
    unique_value = sorted(unique_value)

    "得到每类对象的数量"
    unique_count = [obs_num.count(i) for i in unique_value]

    "类别名称为x轴，对应的数量为y轴"
    array_x = unique_value
    array_y = unique_count

    wh_dataframe = pd.DataFrame(array_y, index=array_x, columns=["obs num"])
    wh_dataframe.plot(kind='bar', color="#55aacc")
    plt.show()
