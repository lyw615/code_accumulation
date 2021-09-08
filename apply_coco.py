import sys, os

"专门用于调用处理coco数据相关的脚本"
file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.data_analyse.data_analyse_coco import analyse_obs_size_after_resized, analyse_obs_ratio,check_annos

# analyse_obs_size_after_resized(r"G:\hrsc\out_coco\train.json", [(640, 480)])
# analyse_obs_ratio(r"G:\hrsc\out_coco\train.json")
check_annos(r"G:\hrsc\out_coco\train.json")