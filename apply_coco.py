import sys, os

"专门用于调用处理coco数据相关的脚本"
file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.data_analyse.data_analyse_coco import analyse_obs_size_after_resized, analyse_obs_ratio,analyse_num_each_class

json_path=r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/104_tv39_hrsc_raw_trans.json"
# analyse_obs_size_after_resized(json_path, [(640, 480)])
analyse_num_each_class(json_path)
# analyse_obs_ratio(r"G:\hrsc\out_coco\train.json")
