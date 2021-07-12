import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.data_analyse.data_analyse import analyse_image_wh

analyse_image_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations", [], plot_type='histogram')
