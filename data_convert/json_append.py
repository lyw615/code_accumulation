import json,os
import numpy as np

category = {1: "echinus", 2: "scallop", 3: "starfish", 4: "holothurian",5:"waterweeds"}
added_json_path=r'/home/data1/yw/data/iobjectspy_out/mmdetection/history_test_result/no_influence/pesual_lowLr_train.bbox.json'
raw_json_path=r'/home/data1/yw/water_detection/split_folds/fold_v1/train.json'
added_json_image=r'/home/data1/yw/water_detection/split_folds/fold_v1/test.json'
new_json_path=r'/home/data1/yw/water_detection/split_folds/fold_v1/train_pseudo_07.json'

with open(added_json_image) as f :
    add_image_jsRecord=f.readlines()
    add_image_jsRecord=eval(add_image_jsRecord[0])

#parse json
with open(added_json_path) as f :
    add_jsRecord=f.readlines()
    add_jsRecord=eval(add_jsRecord[0])



with open(raw_json_path, "r", encoding='utf-8') as f:
    json_dict = json.load(f)

last_image_id=int(json_dict['images'][-1]['id'])
last_anno_id=int(json_dict['annotations'][-1]['id'])

# image field add
add_image = add_image_jsRecord['images']
for img in add_image:
    img['id']+=last_image_id
    json_dict["images"].append(img)



anno_id=last_anno_id+1
for line in add_jsRecord:
    #ann add
    ann = {
        "area": line['bbox'][2]*line['bbox'][3],
        "iscrowd": 0,
        "image_id": line['image_id']+last_image_id,
        "bbox": line['bbox'],
        "category_id": line['category_id'],
        "id": anno_id,
        "ignore": 0,
        "segmentation": [],
    }
    json_dict["annotations"].append(ann)

    anno_id+=1

with open(new_json_path,'w') as f:
    json_str = json.dumps(json_dict)
    f.write(json_str)