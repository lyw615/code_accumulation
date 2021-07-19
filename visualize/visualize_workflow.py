import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..")))

from code_aculat.data_analyse.data_analyse import analyse_image_wh, output_big_wh, analyse_obs_per_image, \
    analyse_obs_w2h_ratio
from code_aculat.visualize.visualize_voc_xml import visualize_voc_xml


# analyse_image_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations", [], plot_type='points')
# analyse_obs_per_image(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations")
# analyse_obs_w2h_ratio(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations")

def visual_big_label():
    # 叠加label可视化尺寸大的数据
    xml_dir = r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations"
    image_dir = r"/home/data1/GeoAI_Data/compete_unzip/train/TIFFImages"
    xmls = output_big_wh(xml_dir, 20000)  # 宽或高大于
    xmls = [os.path.join(xml_dir, x) for x in xmls]

    visualize_voc_xml(xmls,
                      image_dir,
                      image_suffix='.tif',
                      show_pro=1
                      )
