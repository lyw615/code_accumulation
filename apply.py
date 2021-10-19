import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))


from code_aculat.data_analyse.data_analyse import analyse_image_wh, output_big_wh, analyse_obs_scale,\
    analyse_obs_wh,find_empty_xml,analyse_obs_size_after_resized
from code_aculat.visualize.visualize_voc_xml import visualize_voc_xml,visual_large_tif_label


xml_dir=r"H:\data_saved_dir\hangdao_data\Annotations"
# xml_dir=r"/home/data1/yw/data/compt_data/qzb_data/HRSC_data/first_select_Annotations"

# find_empty_xml(xml_dir)

# analyse_image_wh(xml_dir, [], plot_type='points')


visualize_voc_xml(xml_dir,
                  r"H:\data_saved_dir\hangdao_data\Images",
                  image_suffix='.jpg',

                  show_pro=1
                  )

# output_big_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations",10000)

# analyse_obs_scale(xml_dir)
# analyse_obs_size_after_resized(xml_dir,[(4000,800)])
# analyse_obs_wh(xml_dir)

# visual_large_tif_label(
#     xml_source=r"/home/data1/yw/github_projects/personal_github/code_aculat/data_operation/image_pre_none.txt",
#     image_dir=r"/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold-v1/fold_v4/Images",
#     out_dir=r"./pre_none",


#     xml_dir=r"/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/train/Annotations")

