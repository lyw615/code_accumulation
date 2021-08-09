import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))


from code_aculat.data_analyse.data_analyse import analyse_image_wh, output_big_wh, analyse_obs_scale, analyse_obs_wh
from code_aculat.visualize.visualize_voc_xml import visualize_voc_xml,visual_large_tif_label

# analyse_image_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations", [], plot_type='points')


# visualize_voc_xml(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations",
#                   r"/home/data1/GeoAI_Data/compete_unzip/train/TIFFImages",
#                   image_suffix='.tif',

#                   show_pro=0.02
#                   )

# output_big_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations",10000)

# analyse_obs_scale(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations")
# analyse_obs_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations")

visual_large_tif_label(
    xml_source=r"/home/data1/yw/github_projects/personal_github/code_aculat/data_operation/image_pre_none.txt",
    image_dir=r"/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold/fold_v4/Images",
    out_dir=r"./pre_none",
    xml_dir=r"/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/train/Annotations")

