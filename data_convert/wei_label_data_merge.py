import json
import os

jf_base = json.load(
    open("/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold-v1/fold_v2/train.json", "r"))
jf_add = json.load(open("/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold-v2/fold_v1/31.json", "r"))
jf_add_anno = json.load(
    open('/home/data1/yw/data/iobjectspy_out/mmdetection/history_test_result/xf_result/xf_600_300_clean-f2.bbox.json'))
filename2id_base = {}
filename2anno_base = {}

filename2id_add = {}
filename2anno_add = {}

for _dict in jf_base['images']:
    file_name = _dict['file_name']
    filename2id_base[file_name] = _dict['id']

id2filename_base = dict(zip(filename2id_base.values(), filename2id_base.keys()))

for _dict in jf_base['annotations']:
    if id2filename_base[_dict['image_id']] not in filename2anno_base.keys():
        filename2anno_base[id2filename_base[_dict['image_id']]] = []
    filename2anno_base[id2filename_base[_dict['image_id']]].append(_dict)

for _dict in jf_add['images']:
    file_name = _dict['file_name']
    filename2id_add[file_name] = _dict['id']

id2filename_add = dict(zip(filename2id_add.values(), filename2id_add.keys()))

new_json_id = 0
for _dict in jf_add_anno:
    if id2filename_add[_dict['image_id']] not in filename2anno_add.keys():
        filename2anno_add[id2filename_add[_dict['image_id']]] = []

    area = int(_dict['bbox'][2] * _dict['bbox'][3])
    image_id = filename2id_base[id2filename_add[_dict['image_id']]]
    bbox = list(map(int, _dict['bbox']))

    coco_format_dict = {'area': area, 'iscrowd': 0, 'image_id': image_id, 'bbox': bbox,
                        'category_id': _dict['category_id'], 'id': new_json_id, 'ignore': 0, 'segmentation': []}
    filename2anno_add[id2filename_add[_dict['image_id']]].append(coco_format_dict)
    new_json_id += 1

for filename in filename2anno_add.keys():
    if filename in filename2anno_base.keys():
        filename2anno_base[filename] = filename2anno_add[filename]  # replace annotations

dropped_img = []
for filename in filename2id_add:
    if filename not in filename2anno_add.keys():
        dropped_img.append(filename)

# drop no pre from json base
for img_dict in jf_base['images']:
    if img_dict['file_name'] in dropped_img:
        jf_base['images'].remove(img_dict)
        del filename2anno_base[img_dict['file_name']]

# save to json
annotations = []
for filename in filename2anno_base.keys():
    anno = filename2anno_base[filename]

print("ok")
