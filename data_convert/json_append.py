import json, os
import numpy as np
import argparse

"对类名编号"
category = {1: "echinus", 2: "scallop", 3: "starfish", 4: "holothurian", 5: "waterweeds"}


def json_append(added_json_path, raw_json_path, added_json_image, new_json_path):
    with open(added_json_image) as f:
        add_image_jsRecord = f.readlines()
        add_image_jsRecord = eval(add_image_jsRecord[0])

    # parse json
    with open(added_json_path) as f:
        add_jsRecord = f.readlines()
        add_jsRecord = eval(add_jsRecord[0])

    with open(raw_json_path, "r", encoding='utf-8') as f:
        json_dict = json.load(f)

    last_image_id = int(json_dict['images'][-1]['id'])
    last_anno_id = int(json_dict['annotations'][-1]['id'])

    # image field add
    add_image = add_image_jsRecord['images']
    for img in add_image:
        img['id'] += last_image_id
        json_dict["images"].append(img)

    anno_id = last_anno_id + 1
    for line in add_jsRecord:
        # ann add
        ann = {
            "area": line['bbox'][2] * line['bbox'][3],
            "iscrowd": 0,
            "image_id": line['image_id'] + last_image_id,
            "bbox": line['bbox'],
            "category_id": line['category_id'],
            "id": anno_id,
            "ignore": 0,
            "segmentation": [],
        }
        json_dict["annotations"].append(ann)

        anno_id += 1

    with open(new_json_path, 'w') as f:
        json_str = json.dumps(json_dict)
        f.write(json_str)

    print('file saved in %s' % new_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    "要加进去的json文件，带标签"
    parser.add_argument('added_json_path')
    parser.add_argument('raw_json_path')
    "added_json对应的image_id部分"
    parser.add_argument('added_json_image')
    parser.add_argument('new_json_path')

    arg = parser.parse_args()

    json_append(category, arg.added_json_path, arg.raw_json_path, arg.added_json_image, arg.new_json_path)
