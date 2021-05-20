import os, shutil
import json
from tqdm import tqdm
import argparse

"""
将coco数据转成YOLO的txt label标签，并且如果提供了图片的存储位置，会一并将images文件夹创建，完成图片的准备
"""
parser = argparse.ArgumentParser()
parser.add_argument('--json_path',
                    default='/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v2/test.json', type=str,
                    help="input: coco format(json)")
parser.add_argument('--save_path',
                    default='/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v2/yolo_train/val/labels',
                    type=str, help="specify where to save the output dir of labels")
parser.add_argument('--images_source_dir', default=None, type=str, help="where to copy image for images dir")
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
    "检测YOLO的label文件夹里有没有空的txt文件"
    txt_dir = r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v1/yolo_txt_train"
    txt_files = os.listdir(txt_dir)

    for txt in txt_files:
        with open(os.path.join(txt_dir, txt), "r", encoding="utf-8") as f:
            record = f.readlines()
        if len(record) < 1:
            os.remove(os.path.join(txt_dir, txt))
            print("file %s removed " % txt)


def cp_img_from_txtName(txt_dir, out_image_dir, input_image_dir):
    "分别是YOLO的labels文件夹和images文件夹,以及从哪里复制image的路径"

    txt_files = os.listdir(txt_dir)

    for txt in txt_files:
        image_name = txt.replace('.txt', '.jpg')
        image_path = os.path.join(input_image_dir, image_name)

        out_image_path = image_path.replace(input_image_dir, out_image_dir)

        if os.path.exists(image_path):
            shutil.copy(image_path, out_image_path)
        else:
            print('there is no image corespond to the txt {}'.format(txt))


if __name__ == '__main__':
    json_file = arg.json_path  # coco格式的json文件路径
    ana_txt_save_path = arg.save_path  # 存放解析出来的txt文件路径
    images_source_dir = arg.images_source_dir

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    id_map = {}
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:

        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

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

    if images_source_dir:
        images_dir = os.path.join(
            os.path.dirname(ana_txt_save_path), 'images'
        )
        os.makedirs(images_dir, exist_ok=True)
        cp_img_from_txtName(ana_txt_save_path, images_dir, images_source_dir)
