import sys, os

"专门用于调用处理coco数据相关的脚本"
file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.data_analyse.data_analyse_coco import analyse_obs_size_after_resized, analyse_obs_ratio,check_annos,analyse_image_hw,analyse_obs_size

json_path=r"/home/data1/yw/copy_paste_empty/500_aug/merged.json"
# analyse_obs_size_after_resized(r"G:\hrsc\out_coco\train.json", [(640, 480)])
# analyse_obs_ratio(json_path)
# check_annos(r"G:\hrsc\out_coco\train.json")
# analyse_image_hw(json_path)
# analyse_obs_size(json_path)
