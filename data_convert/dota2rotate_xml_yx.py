import os, sys, codecs, rasterio, json
import numpy as np
from tqdm import tqdm


def main():
    save_dir = r"/home/data1/yw/data/iobjectspy_out/test_rotate_label/Annotations1"
    txt_dir = r"/home/data1/yw/data/iobjectspy_out/test_rotate_label/txt"
    txt_dir = r"/home/data1/competition/data/all_ship/rotation_rectangle/labelTxt"
    image_dir = r"/home/data1/yw/copy_paste_empty/500_aug/out_toge"
    image_suffix = ".bmp"
    json_dir = "/home/data1/yw/copy_paste_empty/500_aug/big_coco_split/fold_v1"

    if not json_dir:

        txt_list = os.listdir(txt_dir)
        output(txt_list, txt_dir, image_dir, image_suffix, save_dir)
    else:
        # from train.json test.json get txt_list
        train_list = get_txt_list_from_json(os.path.join(json_dir, "train.json"))
        test_list = get_txt_list_from_json(os.path.join(json_dir, "test.json"))

        output(train_list, txt_dir, image_dir, image_suffix, os.path.join(save_dir, "train"))
        output(test_list, txt_dir, image_dir, image_suffix, os.path.join(save_dir, "test"))


def get_txt_list_from_json(json_path):
    with open(json_path, 'r') as f:
        jf = json.load(f)

    txt_list = []
    for img in jf['images']:
        txt_list.append(img['file_name'].split('.')[0] + '.txt')

    return txt_list


def output(txt_list, txt_dir, image_dir, image_suffix, save_dir):
    "从dota的txt文件中得到rotate det的xml旋转标注"
    os.makedirs(save_dir)
    total_txt_num = len(txt_list)
    saved_xml_num = 0
    not_exist_txt_num = 0

    for txt in tqdm(txt_list):

        if not os.path.exists(os.path.join(txt_dir, txt)):
            not_exist_txt_num += 1
            continue

        with open(os.path.join(txt_dir, txt), 'r') as f:
            records = f.readlines()

        with rasterio.open(os.path.join(image_dir, txt.split('.')[0] + image_suffix)) as ds:
            height, width = ds.height, ds.width

        if len(records) < 1:
            continue

        xml_path = os.path.join(save_dir, txt.split('.')[0] + ".xml")
        with codecs.open(xml_path, "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'VOC' + '</folder>\n')
            xml.write('\t<filename>' + txt.split('.')[0] + image_suffix + '</filename>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(3) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')

            for record in records:
                values = record.strip('\n').split(" ")
                rect = np.array(values[:-2], dtype=np.int)
                rect = np.reshape(rect, (-1, 2))

                if min(rect[:, 0]) != rect[0][0] or min(rect[:, 1]) != rect[1][1] or max(rect[:, 0]) != rect[2][
                    0] or max(rect[:, 1]) != rect[3][1]:
                    raise ("rectangle coordinate problem")

                else:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + values[-2] + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>0</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<x0>' + str(int(round(rect[0][0]))) + '</x0>\n')
                    xml.write('\t\t\t<y0>' + str(int(round(rect[0][1]))) + '</y0>\n')
                    xml.write('\t\t\t<x1>' + str(int(round(rect[1][0]))) + '</x1>\n')
                    xml.write('\t\t\t<y1>' + str(int(round(rect[1][1]))) + '</y1>\n')
                    xml.write('\t\t\t<x2>' + str(int(round(rect[2][0]))) + '</x2>\n')
                    xml.write('\t\t\t<y2>' + str(int(round(rect[2][1]))) + '</y2>\n')
                    xml.write('\t\t\t<x3>' + str(int(round(rect[3][0]))) + '</x3>\n')
                    xml.write('\t\t\t<y3>' + str(int(round(rect[3][1]))) + '</y3>\n')

                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')

            xml.write('</annotation>')
            saved_xml_num += 1

    print("total_txt_num is  %d ,saved_xml_num is %d , not_exist_txt_num is  %d" % (
        total_txt_num, saved_xml_num, not_exist_txt_num))


main()
