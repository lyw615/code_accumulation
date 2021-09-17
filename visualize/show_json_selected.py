import sys, os, json, tqdm

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..")))
from code_aculat.visualize.visual_base import draw_bboxes_on_image, show_two_image
from PIL import Image
import numpy as np


def get_image_all_ann(json_file):
    # 得到图片下所有标签
    image_id2name = {}
    image_name2annos = {}

    for image in json_file['images']:
        image_id2name[image['id']] = image['file_name']

    for ann in json_file['annotations']:
        if image_id2name[ann['image_id']] not in image_name2annos.keys():
            image_name2annos[image_id2name[ann['image_id']]] = []
        image_name2annos[image_id2name[ann['image_id']]].append(ann)

    return image_name2annos


image_dir = r"/home/data1/yw/data/compt_data/qzb_data/HRSC_data/Images"
target_list = os.listdir("/home/data1/yw/data/compt_data/qzb_data/HRSC_data/first_select_Annotations")
target_list = [x.split(".")[0] for x in target_list]

target_catid = [10, 11, 12, 15]
save_dir = "/home/data1/yw/data/compt_data/qzb_data/HRSC_data/visual_unlabel"
os.makedirs(save_dir, exist_ok=True)

js = json.load(open("/home/data1/yw/data/compt_data/qzb_data/HRSC_data/merged_pre.json", 'r'))
image_name2annos = get_image_all_ann(js)

for image in tqdm.tqdm(image_name2annos.keys()):
    if image.split(".")[0] not in target_list:
        # extract bbox ,class_id
        image_bbox = []
        for ann in image_name2annos[image]:
            xmin, ymin, w, h = ann['bbox']
            class_id = ann['category_id']

            if class_id not in target_catid:
                image_bbox.append([xmin, ymin, w, h, class_id])

        if len(image_bbox) > 0:
            image_bbox = np.array(image_bbox)
            image_bbox[:, 2] += image_bbox[:, 0]
            image_bbox[:, 3] += image_bbox[:, 1]
            image_bbox = np.array(image_bbox, dtype=np.int)

            image_show = os.path.join(image_dir, image)
            draw_bboxes_on_image(image_show, image_bbox[:, :4], image_bbox[:, 4], save_dir=save_dir,
                                 save_name=image.split('.')[0] + ".png")
