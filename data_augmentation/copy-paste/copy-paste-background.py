"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

from PIL import Image
import imgviz
import cv2
import argparse
import os, sys, json
import numpy as np
import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast
from pycocotools.coco import COCO
from skimage import measure
from pycocotools import mask as maskUtils

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..", "..")))
from code_aculat.utils.coco_tools import annToMask, MyEncoder


def show_two_image(image1, image2, title=None):
    # 同时可视化两个RGB或者mask
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax1)
    plt.imshow(image1)
    plt.sca(ax2)
    plt.imshow(image2)
    if title:
        plt.title(title)
    plt.show()


def main():
    json_img_dir = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/empty/images/train"
    json_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/empty/annotations/train.json"
    # 背景图片路径
    main_img_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/background"
    imgs_name_main = os.listdir(main_img_path)
    imgs_name_main = list(filter(lambda x: x.endswith(".tif"), imgs_name_main))

    out_img_dir = os.path.join(os.path.dirname(json_img_dir), "new_%s" % os.path.basename(json_img_dir))
    os.makedirs(out_img_dir, exist_ok=True)

    json_save_path = os.path.join(os.path.dirname(json_path), 'new_%s' % os.path.basename(json_path))

    coco = COCO(json_path)
    imgids = coco.getImgIds()

    # main_index=0

    new_images = []
    new_anns = []
    new_cates = []
    new_cate_ids = []
    img_start_id = 1
    ann_start_id = 1
    pasted_index = imgids  # 待粘贴图片索引

    max_copy_num = 57
    for img_index in tqdm.tqdm(pasted_index):
        img_index = 29
        # random choice
        # img_main_name = np.random.choice(imgs_name_main)
        if img_start_id > max_copy_num:
            break
        img_main_name = imgs_name_main[(img_start_id - 1)]
        img_main = cv2.imdecode(
            np.fromfile(os.path.join(main_img_path, img_main_name), dtype=np.uint8), flags=1)

        # get source mask and img
        img_src_info = coco.loadImgs(img_index)[0]
        img_src_path = os.path.join(json_img_dir, img_src_info['file_name'])
        img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), flags=1)

        img_src_height, img_src_width = img_src.shape[:2]
        ann_ids = coco.getAnnIds(imgIds=img_index)
        anns = coco.loadAnns(ann_ids)
        anns_cate_ids = []
        masks = []

        assert len(anns) > 0
        for ann in anns:
            mask_src = annToMask(ann, img_src_height, img_src_width)
            masks.append(mask_src)
            anns_cate_ids.append(ann['category_id'])

            if ann['category_id'] not in new_cate_ids:
                new_cate_ids.append(ann['category_id'])

            break  # 因为添加背景图主要是减少误检率,所以只粘贴一个对象

        if img_src_height > img_main.shape[0] or img_src_width > img_main.shape[1]:
            # resize mask
            img_src, masks = resize_image_mask(img_main.shape[0], img_main.shape[1], img_src, masks)

        alpha = np.zeros(shape=img_src.shape[:2])
        for num_alp in range(len(masks)):
            alpha += masks[num_alp]
        alpha = np.expand_dims(alpha, axis=2)

        # paste image
        img_dtype = img_main.dtype
        row, col = img_src.shape[:2]
        img_main[:row, :col] = img_src * alpha + img_main[:row, :col] * (1 - alpha)
        img_main = img_main.astype(img_dtype)

        # paste mask
        new_main_masks = []
        # check_mask=np.zeros(shape=img_main.shape[:2],dtype=np.uint8)
        for num_mask in range(len(masks)):
            mask = masks[num_mask]
            main_mask = np.zeros(shape=img_main.shape[:2], dtype=np.uint8)
            mask_row, mask_col = mask.shape
            main_mask[:mask_row, :mask_col] = mask
            new_main_masks.append(main_mask)

        #     check_mask+=main_mask
        # show_two_image(img_main,check_mask)

        # create annotation
        new_main_img_path = os.path.join(out_img_dir, "bg_%s" % img_main_name)
        cv2.imencode('.%s' % img_main_name.split('.')[-1], img_main)[1].tofile(new_main_img_path)

        new_images.append(
            {'height': img_main.shape[0], 'width': img_main.shape[1], 'id': img_start_id,
             'file_name': "bg_%s" % img_main_name})

        for num_ann in range(len(new_main_masks)):

            contours = measure.find_contours(new_main_masks[num_ann], 0.5)  # must use np.flip with this method
            if len(contours) > 1:
                new_conts = []
                for cont in contours:
                    if len(cont) > 30:
                        new_conts.append(cont)
                contours = new_conts

            assert len(contours) == 1
            contours = np.flip(contours[0], axis=1)
            segmentation = [contours.ravel().tolist()]
            fortran_bina = np.asfortranarray(new_main_masks[num_ann])
            encoded_bina = maskUtils.encode(fortran_bina)

            bina_bbox = maskUtils.toBbox(encoded_bina).tolist()

            instance_dict = {
                'segmentation': segmentation,  # for polygon  [[x,y,x,y]]
                'iscrowd': 0,
                'image_id': img_start_id,
                'bbox': bina_bbox,
                'area': int(bina_bbox[2] * bina_bbox[3]),
                'category_id': anns_cate_ids[num_ann],
                'id': ann_start_id
            }
            new_anns.append(instance_dict)
            ann_start_id += 1
        img_start_id += 1

    for cate_id in new_cate_ids:
        new_cates.append({'id': cate_id, 'name': str(cate_id)})

    new_coco_json = {'images': new_images, 'annotations': new_anns, 'categories': new_cates}

    with open(json_save_path, 'w') as  f:
        json.dump(new_coco_json, f, cls=MyEncoder)


def resize_image_mask(row, column, image, masks):
    p_row, p_col = image.shape[:2]
    ratio = min(column / p_col, row / p_row) - 0.0001
    resized_image = cv2.resize(image, (int(ratio * p_col), int(ratio * p_row)))  # resize image

    resized_masks = []
    for num_mask in range(len(masks)):  # resize masks
        new_mask = cv2.resize(masks[num_mask], (int(ratio * p_col), int(ratio * p_row)))
        resized_masks.append(new_mask)

    return resized_image, resized_masks


if __name__ == '__main__':
    "从文件中粘贴对象到背景图上"
    main()
