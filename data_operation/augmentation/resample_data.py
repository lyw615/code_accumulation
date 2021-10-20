import os, json
import numpy as np


def stastic_single_class_images(json_path, cate_id):
    if isinstance(json_path, str):
        jf = json.load(open(json_path, 'r'))
    else:
        jf = json_path

    only_cate_imgs = []
    only_imgs_instance_num = 0

    imgid2cate = {}
    for ann in jf['annotations']:
        if ann['image_id'] not in imgid2cate:
            imgid2cate[ann['image_id']] = []

        imgid2cate[ann['image_id']].append(ann['category_id'])

    for key, value in imgid2cate.items():
        unique_cate = list(set(value))
        if len(unique_cate) == 1 and unique_cate[0] == cate_id:
            only_cate_imgs.append(key)
            only_imgs_instance_num += len(value)

    # print(len(only_cate_imgs),only_imgs_instance_num)
    return only_cate_imgs


def down_sample_class():
    json_path = r"H:\resample_data\104_tv39_hrsc_raw_trans_copy.json"
    # json_path=r"H:\bdc10020-7112-468b-801e-bbbc210568f2\train_val\train.json"
    new_json_path = r"H:\resample_data\12_downsample.json"

    cate_id = 12  # 降采样类别id
    save_portion = 0.7  # 降采样比例

    with open(json_path, 'r') as f:
        jf = json.load(f)
    image_ids = stastic_single_class_images(json_path, cate_id)

    saved = image_ids[:int(len(image_ids) * save_portion)]

    # 给当前的image_id 进行降序排列，然后挨个生成新json里image_id的字典
    new_imgid_dict = {}
    new_img_id = 1
    for id in saved:
        if id not in new_imgid_dict:
            new_imgid_dict[id] = new_img_id
            new_img_id += 1

    # instance_id 也要修改
    ins_id = 1
    new_ann = []
    for ann in jf['annotations']:
        if ann['image_id'] in saved:
            ann['id'] = ins_id
            ann['image_id'] = new_imgid_dict[ann['image_id']]
            new_ann.append(ann)
            ins_id += 1

    new_img = []
    for img in jf['images']:
        if img['id'] in saved:
            img['id'] = new_imgid_dict[img['id']]
            new_img.append(img)
    new_cat = []
    new_cat.append([{'name': '12', 'id': cate_id}])

    new_json = {}
    new_json['images'] = new_img
    new_json['annotations'] = new_ann
    new_json['categories'] = new_cat

    json.dump(new_json, open(new_json_path, 'w'))


def main():
    down_sample_class()


main()