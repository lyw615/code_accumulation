from pycocotools.coco import COCO
from tqdm import tqdm
import cv2, os, sys

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..")))

import numpy as np
from code_aculat.visualize.visual_base import show_rotated_bbox_from_txt
from matplotlib import pyplot as plt


def cocotools_visual():
    image_dir = r"/home/data1/competition/data/qzb_test/worldview05m_test/ship_test"
    anno_json_path = r"/home/data1/yw/data/iobjectspy_out/pred_ship_test_1.json"
    # plt_save_dir="/home/data1/yw/data/iobjectspy_out/coco_visual_show"
    # os.makedirs(plt_save_dir)
    coco = COCO(anno_json_path)
    imgIds = coco.getImgIds()
    catIds = coco.getCatIds()

    for imgId in tqdm(imgIds):
        img = coco.loadImgs(imgId)[0]
        file_name = img['file_name']
        image_arr = cv2.imdecode(np.fromfile(os.path.join(image_dir, file_name), dtype=np.uint8), flags=1)
        plt.imshow(image_arr)

        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=0)
        anns = coco.loadAnns(annIds)

        coco.showAnns(anns, draw_bbox=True)

        plt.show()
        # plt.savefig("%s/%s"%(plt_save_dir,file_name.split('.')[0]+'.png'))


def draw_rotated_visual():
    txt_dir = "/home/data1/yw/data/iobjectspy_out/txt_dir_ship_test_1_1_1_1_1_1_1_1_1_1"
    image_dir = "/home/data1/competition/data/qzb_test/worldview05m_test/ship_test"
    visual_saved_dir = "/home/data1/yw/data/iobjectspy_out/visual_save_new_score_iou_train_multiscale"
    os.makedirs(visual_saved_dir, exist_ok=True)
    show_rotated_bbox_from_txt(txt_dir, image_dir, visual_saved_dir)


def main():
    draw_rotated_visual()


main()
