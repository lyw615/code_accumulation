import cv2, os, sys, json
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
from pycocotools import mask as maskUtils
from tqdm import tqdm


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


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)


json_path = r'/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/train.json'
coco = COCO(json_path)
txt_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/train_18.txt"

im_start_ind = 1
instance_start_id = 1
image_out_dir = "/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/new_imgs_18"
os.makedirs(image_out_dir, exist_ok=True)
copy_num = 200
long_class_id = 12

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
], bbox_params=A.BboxParams(format="coco", min_visibility=0.8)
)

data = CocoDetectionCP(
    r'/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/Images', json_path, transform
)

json_save_path = os.path.join(os.path.dirname(json_path), 'new_%s' % os.path.basename(json_path))
image_list = []
ann_list = []

catIds = coco.getCatIds()
for n in tqdm(range(copy_num)):

    loop_num = 0
    # 这里选一张实例数量比较少的尾类图片,经过统计，选3
    while True:
        index = copy_indexs[random.randint(0, len(copy_indexs) - 1)]
        img_info = coco.loadImgs(index)[0]
        if img_info['id'] != index:  # 判断index和image_id是否一致
            raise ("error index {} with image_id {}".format(index, img_info['id']))

        annIds = coco.getAnnIds(imgIds=img_info['id'])
        anns = np.array([ann['category_id'] for ann in coco.loadAnns(annIds)], dtype=np.int)
        if len(annIds) < 5 and len(annIds) > 0 and len(np.where(anns == long_class_id)[0]) < 2:
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
    new_image_name = "cp_%d.jpg" % (im_start_ind + n)
    # cv2.imencode only operates on bgr format image
    cv2.imencode('.%s' % new_image_name.split('.')[-1], image[..., ::-1])[1].tofile(
        os.path.join(image_out_dir, new_image_name))

    image_list.append(
        {'height': image.shape[0], 'width': image.shape[1], 'id': im_start_ind + n, 'file_name': new_image_name})
    from skimage import measure

    for instance_id in range(len(bboxes)):
        contours = measure.find_contours(masks[instance_id], 0.5)  # must use np.flip with this method
        fortran_bina = np.asfortranarray(masks[instance_id])
        encoded_bina = maskUtils.encode(fortran_bina)
        bina_area = int(maskUtils.area(encoded_bina))  # more accuracy
        bina_bbox = maskUtils.toBbox(encoded_bina)

        if len(contours) > 1:
            polygons = []
            for contour in contours:
                if len(contour) < 50:
                    continue
                contour = np.flip(contour, axis=1)
                polygons.append(contour.ravel().tolist())

            mask_row, mask_col = masks[instance_id].shape[:2]
            rles = maskUtils.frPyObjects(polygons, mask_row, mask_col)
            segmentation = maskUtils.merge(rles)

            # #convert rle to bina-mask
            # conv_mask=maskUtils.decode(segmentation)
            # show_two_image(masks[instance_id],conv_mask)
        else:
            try:
                contours = np.flip(contours[0], axis=1)  # must be contours[0], 为了让坐标排列顺序由y,x,y,x变成x,y,x,y
            except Exception as e:
                show_two_image(masks[instance_id], masks[instance_id])
                raise e
            segmentation = [contours.ravel().tolist()]  # polygon是 list, rle不是list

        instance_dict = {
            'segmentation': segmentation,  # for polygon  [[x,y,x,y]]
            'iscrowd': 0,
            'image_id': im_start_ind + n,
            'bbox': bina_bbox.tolist(),
            'area': bina_area,
            'category_id': bboxes[instance_id][-2],
            'id': instance_start_id
        }
        ann_list.append(instance_dict)
        instance_start_id += 1

    # mask_zero = np.zeros(shape=image.shape[:2])
    # for i in range(len(masks)):
    #     mask_zero += masks[i]
    # show_two_image(image, mask_zero)
    # print("ok")

with open(json_path, 'r') as f:
    jf = json.load(f)
    cate_list = jf['categories']

new_coco_json = {'images': image_list, 'annotations': ann_list, 'categories': cate_list}

with open(json_save_path, 'w') as  f:
    json.dump(new_coco_json, f, cls=MyEncoder)
