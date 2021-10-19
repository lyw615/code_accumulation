import os,json
import numpy as np

json_path=r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/104_tv39_hrsc_raw_trans.json"
jf=json.load(open(json_path,'r'))

cate_id=12
all_cate_num=0
only_cate_imgs=[]
only_imgs_instance_num=0

imgid2cate={}
for ann in jf['annotations']:
    if ann['image_id'] not in imgid2cate:
        imgid2cate[ann['image_id']]=[]

    imgid2cate[ann['image_id']].append(ann['category_id'])

for key,value in imgid2cate.items():
    unique_cate=list(set(value))
    if len(unique_cate)==1 and unique_cate[0]==cate_id:
        only_cate_imgs.append(key)
        only_imgs_instance_num+=len(value)

print(len(only_cate_imgs),only_imgs_instance_num)
