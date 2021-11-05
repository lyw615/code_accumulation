from pycocotools.coco import COCO
from tqdm import tqdm
import cv2, os, sys

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..")))

import numpy as np
from code_aculat.visualize.visual_base import show_rotated_bbox_from_txt
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils


def cocotools_visual():
    image_dir = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/new_imgs_18"
    anno_json_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/new_train.json"
    # plt_save_dir="/home/data1/yw/data/iobjectspy_out/coco_visual_show"
    # os.makedirs(plt_save_dir)
    coco = COCO(anno_json_path)
    imgIds = coco.getImgIds()
    catIds = coco.getCatIds()

    for imgId in tqdm(imgIds):
        if np.random.randint(1, 10) < 6:
            continue
        img = coco.loadImgs(imgId)[0]
        file_name = img['file_name']
        image_arr = cv2.imdecode(np.fromfile(os.path.join(image_dir, file_name), dtype=np.uint8), flags=1)
        plt.imshow(image_arr)

        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=0)
        anns = coco.loadAnns(annIds)

        coco.showAnns(anns, draw_bbox=True)

        plt.show()

        # draw_ann_masks(anns, img['height'], img['width'])     #用于检查mask

        # plt.savefig("%s/%s"%(plt_save_dir,file_name.split('.')[0]+'.png'))


def draw_ann_masks(anns, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    return annotation's segmentation to masks
    """
    mask_zero = np.zeros(shape=(height, width))
    for ann in anns:
        mask = annToMask(ann, height, width)
        mask_zero += mask
    plt.imshow(mask_zero)
    plt.show()


def annToRLE(ann, height, width):
    """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # RLE
        rle = ann['segmentation']
    return rle


def annToMask(ann, height, width):
    """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to
        binary mask.
        :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


def draw_rotated_visual():
    txt_dir = "/home/data1/yw/data/iobjectspy_out/txt_dir_ship_test_1_1_1_1_1_1_1_1_1_1"
    image_dir = "/home/data1/competition/data/qzb_test/worldview05m_test/ship_test"
    visual_saved_dir = "/home/data1/yw/data/iobjectspy_out/visual_save_new_score_iou_train_multiscale"
    os.makedirs(visual_saved_dir, exist_ok=True)
    show_rotated_bbox_from_txt(txt_dir, image_dir, visual_saved_dir)


def main():
    # draw_rotated_visual()
    cocotools_visual()


main()
