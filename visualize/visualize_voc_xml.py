import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))
from code_aculat.utils.xml_process import analyze_xml
from code_aculat.visualize.visual_base import draw_bboxes_on_image

import numpy as np


def visualize_voc_xml(xml_source, image_dir, image_suffix='.tif', show_pro=0.7):
    # xml_source: dir,file,list
    if isinstance(xml_source, str):
        if os.path.isdir(xml_source):
            xmls = os.listdir(xml_source)
            xml_source = [os.path.join(xml_source, xml) for xml in xmls]

    for _ in range(len(xml_source)):
        if np.random.randint(1, 1000) / 1000 < show_pro:  # 1/3的概率会被可视化

            # 从xml获取类别名和bbox坐标
            class_names, rectangles = analyze_xml(xml_source[_])

            # 在图像上绘制
            image_name = os.path.basename(xml_source[_]).replace('.xml', image_suffix)
            image_path = os.path.join(image_dir, image_name)
            draw_bboxes_on_image(image_path, rectangles, class_names)
