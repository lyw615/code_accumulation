import json
import os

mmdet_test_json = "/home/data1/yw/data/compt_data/qzb_data/HRSC_data/test.json"
mmdet_pre_json = '/home/data1/yw/data/iobjectspy_out/mmdetection/history_test_result/xf_result/qzb-hrsc.bbox.json'
out_json_path = "/home/data1/yw/data/compt_data/qzb_data/HRSC_data/merged_pre.json"

jf_add = json.load(open(mmdet_test_json, "r"))
jf_add_anno = json.load(open(mmdet_pre_json, 'r'))

annos = []
instance_id = 0
for _dict in jf_add_anno:
    area = int(_dict['bbox'][2] * _dict['bbox'][3])  # bbox index 2,3 is width height
    image_id = _dict['image_id']
    bbox = list(map(int, _dict['bbox']))

    # get coco format information
    coco_format_dict = {'area': area, 'iscrowd': 0, 'image_id': image_id, 'bbox': bbox,
                        'category_id': _dict['category_id'], 'id': instance_id, 'ignore': 0, 'segmentation': []}

    annos.append(coco_format_dict)
    instance_id += 1

new_json = {}
new_json['images'] = jf_add['images']
new_json['categories'] = jf_add['categories']
new_json['annotations'] = annos

# save to json
json.dump(new_json, open(out_json_path, 'w'))
