import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.data_analyse.data_analyse import analyse_image_wh
from code_aculat.visualize.visualize_voc_xml import  visualize_voc_xml
# analyse_image_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations", [], plot_type='histogram')
visualize_voc_xml(r"E:\push_branch\resources_ml\out\training\object_ext_train_data\Annotations\new_anno",
                  r"E:\push_branch\resources_ml\out\training\object_ext_train_data\new_Images",
                  image_suffix='.tif'
                  )
