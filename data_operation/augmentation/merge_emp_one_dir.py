import os, shutil
import numpy as np
import cv2, json

from skimage import measure


def get_annos_from_mask(mask_path, image_id, cate_id, instance_id):
    """
    从mask文件里得到里面每个对象的annotation
    Args:
        mask_path:
        image_id:
        cate_id:
        instance_id:

    Returns:

    """

    mask_arr = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), flags=1)
    # 避免这个通道的mask没有对象
    ground_truth_binary_mask = mask_arr[:, :, 1]
    if ground_truth_binary_mask.max() < 1:
        ground_truth_binary_mask = mask_arr[:, :, 2]
        if ground_truth_binary_mask.max() < 1:
            ground_truth_binary_mask = mask_arr[:, :, 0]
            if ground_truth_binary_mask.max() < 1:
                raise ("max mask value low than 1 %s %s" % (mask_dir, mask))

    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annos_list = []

    for idx in range(len(contours)):
        contour = contours[idx]
        contour = np.flip(contour, axis=1)

        arr_seg = np.expand_dims(contour, axis=1)
        arr_seg = np.array(arr_seg, dtype=np.int)
        x, y, w, h = cv2.boundingRect(arr_seg)
        segmentation = contour.ravel().tolist()

        annotation = {
            "segmentation": [],
            "area": int(w * h),
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [x, y, x + w, y + h],
            "category_id": cate_id,
            "id": instance_id
        }
        instance_id += 1  # 返回的值是下一个对象的id编号

        annotation["segmentation"].append(segmentation)
        annos_list.append(annotation)

    return annos_list, instance_id, mask_arr.shape


# 自成一个coco json，然后合并两个coco
input_dir = r"/home/data1/yw/copy_paste_empty/500_aug/out_paste"
out_img_dir = r"/home/data1/yw/copy_paste_empty/500_aug/out_toge1"
out_coco_path = r"/home/data1/yw/copy_paste_empty/500_aug/outcoco1.json"
os.makedirs(out_img_dir, exist_ok=True)

class_ids = {}
image_id = 1
instance_id = 1
annotations = []
images = []
json_dict = {}
json_dict["categories"] = []
json_dict["type"] = "instances"
for dirname in os.listdir(input_dir):
    class_num = dirname.split("_")[-1]
    class_ids[class_num] = int(class_num)

for key, value in class_ids.items():  # 把json里的类别字段填好
    json_dict["categories"].append({"id": value, 'name': key})

for dirname in os.listdir(input_dir):
    single_dir = os.path.join(input_dir, dirname)
    class_num = dirname.split("_")[-1]
    class_id = class_ids[class_num]

    img_dir = os.path.join(single_dir, 'JPEGImages')
    mask_dir = os.path.join(single_dir, 'SegmentationClass')

    for mask in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask)
        anno_list, instance_id, mask_shape = get_annos_from_mask(mask_path, image_id=image_id, cate_id=class_id,
                                                                 instance_id=instance_id)

        if len(anno_list) == 0:
            continue

        img_path = os.path.join(img_dir, mask.replace(".png", '.bmp'))

        # 把图片移动到指定文件夹，图片名以paste_emp开头，0开始计数
        new_name = "paste_emp_%d.bmp" % image_id
        shutil.copy(img_path, os.path.join(out_img_dir, new_name))  # 最开始先copy到一个新建文件夹，然后看没有报错，再使用move到指定目录

        # 生成image
        images.append({'height': mask_shape[0], 'width': mask_shape[1], 'id': image_id, 'file_name': new_name})

        annotations += anno_list  # 得到annotation
        image_id += 1

# 形成一个单独的coco文件
json_dict["annotations"] = annotations
json_dict['images'] = images

json.dump(json_dict, open(out_coco_path, 'w'))

# 和之前的coco文件合并
