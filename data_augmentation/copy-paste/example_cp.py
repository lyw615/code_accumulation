import cv2, os, sys
import numpy as np

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.dirname(file_path)))
from copy_paste import CopyPaste
from coco import CocoDetectionCP
from visualize import display_instances
import albumentations as A
import random
from matplotlib import pyplot as plt
from pycocotools.coco import COCO


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


json_path = r'/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/104_tv39_hrsc.json'
coco = COCO(json_path)
txt_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/104_tv39_hrsc.txt"
with open(txt_path, 'r') as f:
    txt_indexs = f.readlines()
txt_indexs = [int(x.strip('\n')) for x in txt_indexs]

all_Imgids = coco.getImgIds()
copy_indexs = [x for x in all_Imgids if x not in txt_indexs]

transform = A.Compose([
    # A.RandomScale(scale_limit=(0.8, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
    # A.PadIfNeeded(256, 256, border_mode=0), #pads with image in the center, not the top left like the paper
    # A.RandomCrop(256, 256),
    CopyPaste(blend=True, sigma=1, pct_objects_paste=1, p=1.)  # 这张图中复制对象的比例
], bbox_params=A.BboxParams(format="coco", min_visibility=1)
)

data = CocoDetectionCP(
    r'/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/Images', json_path, transform
)

copy_num = 5
catIds = coco.getCatIds()
for n in range(copy_num):
    loop_num = 0
    # 这里选一张实例数量比较少的长类图片,经过统计，选3
    while True:
        index = copy_indexs[random.randint(0, len(copy_indexs) - 1)]
        img_info = coco.loadImgs(index)[0]
        if img_info['id'] != index:  # 判断index和image_id是否一致
            raise ("error index {} with image_id {}".format(index, img_info['id']))

        annIds = coco.getAnnIds(imgIds=img_info['id'])
        if len(annIds) < 4 and len(annIds) > 0:
            break

        loop_num += 1
        if loop_num == len(copy_indexs):  # 避免在这里找不到合适的图片，造成死循环
            raise ("there is no target image")

    # image_id 应该是等于index的值的
    img_data = data[index]
    # 这里出来的应该是已经增强过的数据
    image = img_data['image']
    masks = img_data['masks']
    bboxes = img_data['bboxes']

    # 将这些增强后的结果存入新的json

    mask_zero = np.zeros(shape=image.shape[:2])
    for i in range(len(masks)):
        mask_zero += masks[i]
    show_two_image(image, mask_zero)
    print("ok")
