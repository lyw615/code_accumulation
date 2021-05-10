import os,shutil
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v2/test.json', type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v2/yolo_train/val/labels', type=str, help="specify where to save the output dir of labels")
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


def check_txt_empty():
    "check out empty txt file in yolo format"
    txt_dir=r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v1/yolo_txt_train"
    txt_files=os.listdir(txt_dir)

    for txt in txt_files:
        with open(os.path.join(txt_dir,txt),"r",encoding="utf-8") as f :
            record=f.readlines()
        if len(record)<1:
            os.remove(os.path.join(txt_dir,txt))
            print("file %s removed "%txt)


def cp_img_from_txtName():
    txt_dir=r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v2/yolo_train/train/labels"
    out_image_dir=r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v2/yolo_train/train/images"

    txt_files=os.listdir(txt_dir)

    input_image_dir=r"/home/data1/yw/mmdetection/data/water_detection/train/JPEGImages"

    for txt in txt_files:
        image_name=txt.replace('.txt','.jpg')
        image_path=os.path.join(input_image_dir,image_name)

        out_image_path=image_path.replace(input_image_dir,out_image_dir)

        if os.path.exists(image_path):
            shutil.copy(image_path,out_image_path)
        else:
            print('there is no image corespond to the txt {}'.format(txt))


if __name__ == '__main__':
    json_file = arg.json_path  # COCO Object Instance ÀàÐÍµÄ±ê×¢
    ana_txt_save_path = arg.save_path  # ±£´æµÄÂ·¾¶

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

    # cp_img_from_txtName()