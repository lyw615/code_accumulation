import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.data_analyse.data_analyse import analyse_image_wh, output_big_wh,analyse_obs_per_image,analyse_obs_w2h_ratio,analyse_obs_wh
from code_aculat.visualize.visualize_voc_xml import visualize_voc_xml


# visualize_voc_xml(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations",
#                   r"/home/data1/GeoAI_Data/compete_unzip/train/TIFFImages",
#                   image_suffix='.tif',
#                   show_pro=1
#                   )

output_big_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations",20000)
