import sys, os

"专门用于调用处理coco数据相关的脚本"
file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.data_analyse.data_analyse_coco import analyse_obs_size_after_resized, analyse_obs_ratio, check_annos, \
    analyse_image_hw, analyse_obs_size, analyse_num_each_class, stastic_ann_per_image, checkout_iterstrat_split, \
    check_empty_coco

json_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/train_13_18_14_17_16_bg57.json"
# json_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/train_13_18_14_17_16.json"
# analyse_obs_size_after_resized(json_path, [(640, 480)])
# analyse_obs_ratio(json_path)
# check_annos(r"G:\hrsc\out_coco\train.json")
# analyse_image_hw(json_path)
# analyse_obs_size(json_path)
analyse_num_each_class(json_path)
# stastic_ann_per_image(json_path)
# check_empty_coco(json_path)
