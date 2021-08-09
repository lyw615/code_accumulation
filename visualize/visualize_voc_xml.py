import sys, os
from tqdm import tqdm

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))
from code_aculat.utils.xml_process import analyze_xml
from code_aculat.visualize.visual_base import draw_bboxes_on_image, show_bboxes_region

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


def visual_large_tif_label(xml_source, image_dir, out_dir, xml_dir=None):
    """
    可视化大尺寸tif影像上的label区域
    Returns:

    """

    with open(xml_source) as xf:
        content = xf.readlines()

    os.makedirs(out_dir, exist_ok=True)

    for _ in tqdm(range(len(content))):
        tif_name = content[_].strip('\n')
        xml_file = tif_name.split(".")[0] + '.xml'
        xml_file = os.path.join(xml_dir, xml_file)
        class_names, rectangles = analyze_xml(xml_file)

        image_path = os.path.join(image_dir, tif_name)
        show_bboxes_region(image_path, rectangles, class_names, out_dir)
