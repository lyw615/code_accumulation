#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm

START_BOUNDING_BOX_ID = 1
# 预定义类别名称和编号
PRE_DEFINE_CATEGORIES = {"echinus": 1, "scallop": 2, "starfish": 3, "holothurian": 4, "waterweeds": 5}
PRE_DEFINE_CATEGORIES = {"Airport": 1, "Port": 2}


# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_dir, csv_path, json_file):
    if csv_path:
        with open(csv_path, "r") as f:
            lines_record = f.readlines()
            lines_record.pop(0)

        xml_files = []

        for line in lines_record:
            xml_files.append(os.path.join(xml_dir, line.strip("\n").split(',')[1] + ".xml"))

        print("Number of xml files: {}".format(len(xml_files)))
    else:
        xml_files = [os.path.join(xml_dir, file) for file in os.listdir(xml_dir)]

    start_ind = 1
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID

    for xml_file in tqdm(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            # filename = get_and_check(root, "filename", 1).text
            filename = get_and_check(root, "frame", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = start_ind
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)

        if len(filename.split('.')) == 1:
            filename += ".jpg"

        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                # new_id = len(categories)
                # categories[category] = new_id
                continue
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

        start_ind += 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

    print("next time start index%s" % start_ind)


if __name__ == "__main__":
    """
    根据pandas创建的csv文件内的xml文件名，将这些xml转成coco格式的数据，
    需要传入xml文件的存储路径和csv文件的路径
    如果csv_dir是一个文件夹，那说明里面有好几个csv文件都需要做这些的处理
    start_ind 是image_id 的起始 编号
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format."
    )
    parser.add_argument("xml_dir", help="Directory path to xml files.", type=str)
    parser.add_argument("--csv_dir", help="Path to csv directory.", type=str, default=None)

    args = parser.parse_args()

    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    if args.csv_dir == None:
        csv_files = None

    elif os.path.isdir(args.csv_dir):
        csv_files = os.listdir(args.csv_dir)
        csv_files = list(filter(lambda x: x.endswith(".csv"), csv_files))
        csv_files = [os.path.join(args.csv_dir, csv) for csv in csv_files]

    elif os.path.isfile(args.csv_dir):
        csv_files = [args.csv_dir]

    else:
        raise ValueError

    if csv_files:

        for csv_path in csv_files:

            json_file = csv_path.replace(".csv", ".json")

            if os.path.exists(json_file):
                print('file already exist in %s' % json_file)
                raise ValueError

            convert(args.xml_dir, csv_path, json_file)
    else:
        json_file = os.path.join(os.path.dirname(args.xml_dir), "voc2coco.json")
        if os.path.exists(json_file):
            raise ValueError
        convert(args.xml_dir, csv_files, json_file)
    print("Success: {}".format(json_file))
