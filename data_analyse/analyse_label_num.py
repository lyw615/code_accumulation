import pandas as pd
import os
import os.path as osp
csv_path=r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v1/test_right.csv"
df=pd.read_csv(csv_path)
aa=df.values

txt_dir=r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v1/yolo_txt_test"
txt_list=os.listdir(txt_dir)

print("len of aa {} and len of txt_list {}".format(aa.shape[0],len(txt_list)))
for txt in txt_list:
    if txt.strip('.txt') not in aa:
        print(txt)
