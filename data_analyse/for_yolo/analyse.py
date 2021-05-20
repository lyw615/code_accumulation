import pandas as pd
import os
import os.path as osp


def check_txt_not_in_csv_record():
    "查看txt文件夹中哪个文件不在csv的记录之中"
    csv_path = r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v1/test_right.csv"
    txt_dir = r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v1/yolo_txt_test"
    df = pd.read_csv(csv_path)

    # 获取dataframe对象的值赋给aa，而不是以dataframe形式去判断
    aa = df.values

    txt_list = os.listdir(txt_dir)

    print("len of aa {} and len of txt_list {}".format(aa.shape[0], len(txt_list)))
    for txt in txt_list:
        if txt.strip('.txt') not in aa:
            print(txt)
