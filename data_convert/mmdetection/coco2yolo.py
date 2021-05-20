import os
import json
from tqdm import tqdm
import argparse

"""
只需要传入标准的coco格式文件路径和保存YOLO的txt文件路径即可，后者会自动创建
yolo的数据存放格式：
train(训练集下的文件目录结构):
    images
        -image1
        -image2
    labels
        -x1.txt
        -x2.txt
        
"""
parser = argparse.ArgumentParser()
parser.add_argument('--json_path',
                    default='/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v2/test.json', type=str,
                    help="input: coco format(json)")
parser.add_argument('--save_path',
                    default='/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v2/yolo_train/val/labels',
                    type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


if __name__ == '__main__':
    json_file = arg.json_path
    ana_txt_save_path = arg.save_path

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    id_map = {}  # cocoÊý¾Ý¼¯µÄid²»Á¬Ðø£¡ÖØÐÂÓ³ÉäÒ»ÏÂÔÙÊä³ö£¡
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # Ð´Èëclasses.txt
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
    # print(id_map)

    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # ¶ÔÓ¦µÄtxtÃû×Ö£¬ÓëjpgÒ»ÖÂ
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
