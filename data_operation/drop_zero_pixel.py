import cv2
import os, sys, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage import measure
from tqdm import tqdm

"专门用于调用处理coco数据相关的脚本"
file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..")))
from code_aculat.utils.coco_tools import annToMask, MyEncoder


def get_zero_boundry(image):
    row, col = image.shape[:2]
    vertical_line = image[:, int(col / 2), :]
    horizontal_line = image[int(row / 2), :, :]

    v_value = np.where(vertical_line > 0)[0]
    ymin, ymax = v_value[0], v_value[-1]

    h_value = np.where(horizontal_line > 0)[0]
    xmin, xmax = h_value[0], h_value[-1]

    return xmin, xmax, ymin, ymax


def main():
    "去除图片外围的0填充像素,并更新json标注"

    json_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/Json/99.json"
    image_dir = "/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/Images"
    out_image_dir = os.path.join(os.path.dirname(image_dir), "new_%s" % os.path.basename(image_dir))
    os.makedirs(out_image_dir, exist_ok=True)

    json_save_path = os.path.join(os.path.dirname(json_path), "new_%s" % os.path.basename(json_path))
    coco = COCO(json_path)
    imgids = coco.getImgIds()  # 获取所有的image id

    new_images = []
    new_anns = []

    for imgid in tqdm(imgids):
        image_info = coco.loadImgs(imgid)[0]
        image_path = os.path.join(image_dir, image_info['file_name'])

        # process image array
        img_arr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        xmin, xmax, ymin, ymax = get_zero_boundry(img_arr)
        img_arr = img_arr[ymin:ymax + 1, xmin:xmax + 1]

        # process image info
        old_height = image_info['height']
        old_width = image_info['width']

        image_info['height'] = img_arr.shape[0]
        image_info['width'] = img_arr.shape[1]

        new_images.append(image_info)
        cv2.imencode(".%s" % image_info['file_name'].split('.')[-1], img_arr)[1].tofile(
            os.path.join(out_image_dir, image_info['file_name']))

        ann_ids = coco.get_ann_ids(img_ids=image_info['id'])
        anns = coco.loadAnns(ann_ids)

        # update annotaions
        for ann in anns:
            new_mask = annToMask(ann, old_height, old_width)
            new_mask = new_mask[ymin:ymax + 1, xmin:xmax + 1]
            contours = measure.find_contours(new_mask, 0.5)  # must use np.flip with this method

            if len(contours) > 1:
                new_cont = []
                for ic in range(len(contours)):
                    if len(contours[ic]) > 20:
                        new_cont.append(contours[ic])
                contours = new_cont

            assert len(contours) == 1
            contours = np.flip(contours[0], axis=1)
            segmentation = [contours.ravel().tolist()]

            fortran_bina = np.asfortranarray(new_mask)
            encoded_bina = maskUtils.encode(fortran_bina)

            bina_bbox = maskUtils.toBbox(encoded_bina)

            ann["segmentation"] = segmentation
            ann["bbox"] = bina_bbox.tolist()

            new_anns.append(ann)

    with open(json_path, 'r') as f:
        jf = json.load(f)
        cate_list = jf['categories']

    new_coco_json = {'images': new_images, 'annotations': new_anns, 'categories': cate_list}

    with open(json_save_path, 'w') as  f:
        json.dump(new_coco_json, f, cls=MyEncoder)
