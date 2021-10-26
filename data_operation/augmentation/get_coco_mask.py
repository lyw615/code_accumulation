"""
get semantic segmentation annotations from coco data set.
"""
from PIL import Image
import imgviz
import argparse
import os
import tqdm
import numpy as np
import shutil
from pycocotools.coco import COCO


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def main(args):
    annotation_file = os.path.join(args.input_dir, 'annotations', '{}.json'.format(args.split))
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'JPEGImages'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
            img_origin_path = os.path.join(args.input_dir, 'images', args.split, img['file_name'])
            img_output_path = os.path.join(args.input_dir, 'JPEGImages', img['file_name'])
            seg_output_path = os.path.join(args.input_dir, 'SegmentationClass',
                                           img['file_name'].replace('.jpg', '.png'))
            shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(mask, seg_output_path)


def modi_main(args):
    image_dir = r"/home/data1/yw/copy_paste_empty/outship_cls_num"
    for dirname in os.listdir(args.input_dir):
        input_dir = os.path.join(args.input_dir, dirname)
        single_img_dir = os.path.join(image_dir, dirname, dirname.split("_")[-1])
        annotation_file = os.path.join(input_dir, '{}.json'.format(args.split))
        os.makedirs(os.path.join(input_dir, 'SegmentationClass'), exist_ok=True)
        os.makedirs(os.path.join(input_dir, 'JPEGImages'), exist_ok=True)
        coco = COCO(annotation_file)
        catIds = coco.getCatIds()
        imgIds = coco.getImgIds()
        print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
        for imgId in tqdm.tqdm(imgIds, ncols=100):
            img = coco.loadImgs(imgId)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            if len(annIds) > 0:
                mask = coco.annToMask(anns[0]) * anns[0]['category_id']
                for i in range(len(anns) - 1):
                    mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
                img_origin_path = os.path.join(single_img_dir, img['file_name'])
                img_output_path = os.path.join(input_dir, 'JPEGImages', img['file_name'])
                seg_output_path = os.path.join(input_dir, 'SegmentationClass',
                                               img['file_name'].replace('.jpg', '.png'))
                shutil.copy(img_origin_path, img_output_path)
                save_colored_mask(mask, seg_output_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=r"H:\resample_data", type=str,
                        help="coco dataset directory")
    parser.add_argument("--split", default="train", type=str,
                        help="train2017 or val2017")
    return parser.parse_args()


if __name__ == '__main__':
    # 从coco的点集里解析出mask，将多个mask以非零数组的方式叠加到图上,即生成VOC格式的mask
    args = get_args()
    main(args)
    # modi_main(args)
