import os, sys, json

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..", "..")))
from code_aculat.visualize.visual_base import show_two_image

import numpy as np
import cv2


def stastic_single_class_images(json_path, cate_id):
    if isinstance(json_path, str):
        jf = json.load(open(json_path, 'r'))
    else:
        jf = json_path

    only_cate_imgs = []
    only_imgs_instance_num = 0

    imgid2cate = {}
    for ann in jf['annotations']:
        if ann['image_id'] not in imgid2cate:
            imgid2cate[ann['image_id']] = []

        imgid2cate[ann['image_id']].append(ann['category_id'])

    for key, value in imgid2cate.items():
        unique_cate = list(set(value))
        if len(unique_cate) == 1 and unique_cate[0] == cate_id:
            only_cate_imgs.append(key)
            only_imgs_instance_num += len(value)

    print(len(only_cate_imgs), only_imgs_instance_num)
    return only_cate_imgs


def down_sample_single_class():
    json_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/99_tv37_hrsc605.json"
    # json_path=r"H:\bdc10020-7112-468b-801e-bbbc210568f2\train_val\train.json"
    new_json_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/99_tv37_hrsc605_12_downsample.json"

    cate_id = 12  # 降采样类别id
    drop_portion = 1  # 降采样比例

    with open(json_path, 'r') as f:
        jf = json.load(f)
    image_ids = stastic_single_class_images(json_path, cate_id)

    drop_imgids = image_ids[:int(len(image_ids) * drop_portion)]

    # create_new_json_from_imageid(saved, jf, new_json_path)
    drop_image_and_save_json(drop_imgids, jf, new_json_path)


def create_new_json_from_imageid(image_id, jf, new_json_path):
    # 给当前的image_id 进行降序排列，然后挨个生成新json里image_id的字典
    new_imgid_dict = {}
    new_img_id = 1
    for id in image_id:
        if id not in new_imgid_dict:
            new_imgid_dict[id] = new_img_id
            new_img_id += 1

    all_cate_id = []
    # instance_id 也要修改
    ins_id = 1
    new_ann = []
    for ann in jf['annotations']:
        if ann['image_id'] in image_id:  # 把所有在目标image_id里的ann都修改id和image_id后添加进新的annotation
            ann['id'] = ins_id
            ann['image_id'] = new_imgid_dict[ann['image_id']]
            new_ann.append(ann)

            if ann['category_id'] not in all_cate_id:
                all_cate_id.append(ann['category_id'])
            ins_id += 1

    new_img = []
    for img in jf['images']:
        if img['id'] in image_id:
            img['id'] = new_imgid_dict[img['id']]
            new_img.append(img)

    new_cat = []  # 这里需要把所有出现的类别都添加上去
    for cat in all_cate_id:
        new_cat.append({'id': cat})

    new_json = {}
    new_json['images'] = new_img
    new_json['annotations'] = new_ann
    new_json['categories'] = new_cat

    json.dump(new_json, open(new_json_path, 'w'))


def drop_image_and_save_json(drop_imgids, jf, new_json_path):
    new_images = []
    new_anns = []
    new_cate = []
    for img in jf['images']:
        if img['id'] not in drop_imgids:
            new_images.append(img)

    for ann in jf['annotations']:
        if ann['image_id'] not in drop_imgids:
            new_anns.append(ann)

    old_imgid2new_dic = {}
    for img, new_id in zip(new_images, range(1, len(new_images) + 1)):
        old_imgid2new_dic[img['id']] = new_id
        img['id'] = new_id

    cate_ids = []
    for ann, ann_id in zip(new_anns, range(1, len(new_anns) + 1)):
        ann['image_id'] = old_imgid2new_dic[ann['image_id']]
        ann['id'] = ann_id
        if ann['category_id'] not in cate_ids:
            cate_ids.append(ann['category_id'])

    for cate in jf['categories']:
        if cate['id'] in cate_ids:
            new_cate.append(cate)

    new_jf = {'images': new_images, 'annotations': new_anns, 'categories': new_cate}
    json.dump(new_jf, open(new_json_path, 'w'))


def resample_single_class():
    json_path = r"H:\resample_data\104_tv39_hrsc_raw_trans_copy.json"
    json_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/train.json"

    cate_id = 18  # 重采样类别id
    new_txt_path = os.path.join(os.path.dirname(json_path),
                                os.path.basename(json_path).split('.')[0] + "_" + str(cate_id) + '.txt')
    with open(json_path, 'r') as f:
        jf = json.load(f)

    image_id = []

    for ann in jf['annotations']:
        if ann['category_id'] == cate_id:
            if ann['image_id'] not in image_id:
                image_id.append(ann['image_id'])

    # 把带有这类目标的image_id都存到另一个txt里面，然后做copy-paste,得到的数据都存在另一个json里
    with open(new_txt_path, 'w') as f:
        for id in image_id:
            f.write("%d\n" % id)

    # create_new_json_from_imageid(image_id, jf, new_json_path)


def resample_aug():
    """
    对VOC格式的数据进行增强
    Returns:

    """
    from albumentations import Rotate, CenterCrop
    import cv2

    # 旋转、像素增强、mosaic增强
    data_dir = r"H:\resample_data"

    image_dir = os.path.join(data_dir, 'JPEGImages')
    seg_dir = os.path.join(data_dir, 'SegmentationClass')

    for img in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img)
        mask_path = os.path.join(seg_dir, img.split('.')[0] + '.png')

        image_arr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        seg_arr = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), 1)

        mask_arr = seg_arr[:, :, 1]  # 避免得到mask里空白的channel
        if mask_arr.max() < 1:
            mask_arr = seg_arr[:, :, 2]
            if mask_arr.max() < 1:
                mask_arr = seg_arr[:, :, 0]
                if mask_arr.max() < 1:
                    raise ("max mask value low than 1 %s" % (mask_path))
        # 多次的旋转增强
        rotate_list = np.random.choice([x * 15 for x in range(1, 20)], 3)
        for degree in rotate_list:
            rota_img = Rotate(limit=(degree, degree), p=1)(image=image_arr)['image']
            rota_mask = Rotate(limit=(degree, degree), p=1)(image=mask_arr)['image']
            show_two_image(rota_img, rota_mask)
            print('ok')


def shift_move():
    image_dir = r'H:\resample_data\JPEGImages'
    out_dir = os.path.join(os.path.dirname(image_dir), "shift_move")
    os.makedirs(out_dir)

    # 左右、上下平移
    # a为平移的尺度,这里设置为10.
    move_stride = 50

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)

        size = img.shape[:2]
        img1 = img
        img2 = img
        img3 = img
        img4 = img
        img1 = np.concatenate((img1[:, move_stride:], img1[:, :move_stride]), axis=1)  # 左

        cv2.imencode('.%s' % img_name.split('.')[-1], img1)[1]. \
            tofile(os.path.join(out_dir, '%s' % (img_name.split('.')[0] + '_left.png')))

        img2 = np.concatenate((img2[:, size[1] - move_stride:], img2[:, :size[1] - move_stride]), axis=1)  # 右
        cv2.imencode('.%s' % img_name.split('.')[-1], img2)[1]. \
            tofile(os.path.join(out_dir, '%s' % (img_name.split('.')[0] + '_right.png')))

        img3 = np.concatenate((img3[move_stride:, :], img3[:move_stride, :]), axis=0)  # 上
        cv2.imencode('.%s' % img_name.split('.')[-1], img3)[1]. \
            tofile(os.path.join(out_dir, '%s' % (img_name.split('.')[0] + '_up.png')))

        img4 = np.concatenate((img4[size[0] - move_stride:, :], img4[:size[0] - move_stride, :]), axis=0)  # 下
        cv2.imencode('.%s' % img_name.split('.')[-1], img4)[1]. \
            tofile(os.path.join(out_dir, '%s' % (img_name.split('.')[0] + '_bottom.png')))
        if np.random.randint(1, 10) > 7:
            show_two_image(img, img1, 'left')
            show_two_image(img, img2, 'right')
            show_two_image(img, img3, 'up')
            show_two_image(img, img4, 'bot')


def main():
    # down_sample_single_class()
    resample_single_class()
    # resample_aug()
    # shift_move()


main()
