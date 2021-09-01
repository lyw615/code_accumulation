"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

from PIL import Image
import imgviz
import cv2
import argparse
import os
import numpy as np
import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def show_one_image(image, title=None, save_dir=None, save_name=None):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)

    if title:
        plt.title(title)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name))
    else:
        plt.show()
    # outdir="/home/data1/yw/github_projects/personal_github/code_aculat/outship"
    # os.makedirs(outdir,exist_ok=True)
    # num=len(os.listdir(outdir))
    # plt.savefig(os.path.join(outdir,"%d.png"%num))


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img


def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)

    if sub_img01.shape != img_main.shape:
        sub_img01 = cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]), interpolation=cv2.INTER_NEAREST)
    # show_two_image(sub_img01,mask)

    if mask.shape != img_main.shape[:2]:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_02 = np.asarray(mask, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    # show_two_image(sub_img02,mask_02)

    img_main = img_main - sub_img02 + sub_img01
    return img_main


def img_add_modi(img_main, rotate_list):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape

    for i in range(len(rotate_list)):
        mask_src, img_src = rotate_list[i]
        mask = np.asarray(mask_src, dtype=np.uint8)
        sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)

        if sub_img01.shape != img_main.shape:
            sub_img01 = cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]), interpolation=cv2.INTER_NEAREST)
        # show_two_image(sub_img01,mask)

        if mask.shape != img_main.shape[:2]:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_02 = np.asarray(mask, dtype=np.uint8)
        sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                            mask=mask_02)
        # show_two_image(sub_img02,mask_02)

        img_main = img_main - sub_img02 + sub_img01

        # 计算已有标签的外接矩形，然后看是否有空间填充更多的标签，而且得保证标签不能在过于边缘的地方
    return img_main


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


def rescale_src(mask_src, img_src, h, w):
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape

    src_shape = img_src.shape
    if src_shape[0] <= 160 and src_shape[1] <= 160:  # 如果是对主办方给的数据进行paste，那么基本相当于对单个mask进行操作，但如果不是单个mask，那么shape一般不会这么小
        rescale_ratio = np.random.uniform(1, 2)

    else:
        max_reshape_ratio = min(h / h_src, w / w_src)

        if max_reshape_ratio > 1:
            min_reshape_ratio = 0.9
            max_reshape_ratio = 1  # 主办方给的图片本来也不会特别小，所以一般不用再进行放大
        else:  # src太大，需要缩小
            min_reshape_ratio = min(0.8, max_reshape_ratio)

        rescale_ratio = np.random.uniform(min_reshape_ratio, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w), dtype=np.uint8)
    try:
        img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    except:
        pass
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


def Large_Scale_Jittering(mask, img, min_scale=0.7, max_scale=1.0):  # 目前主要考虑小目标放大，一些大目标不做此增强
    img_shape = img.shape
    if img_shape[0] <= 120 and img_shape[1] <= 120:
        max_scale = 2.0
        min_scale = 1.3

    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    # x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    # if rescale_ratio <= 1.0:  # padding
    #     img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
    #     mask_pad = np.zeros((h, w), dtype=np.uint8)
    #     img_pad[y:y+h_new, x:x+w_new, :] = img
    #     mask_pad[y:y+h_new, x:x+w_new] = mask
    #
    #     return mask_pad, img_pad
    # else:  # crop
    # img_crop = img[y:y+h, x:x+w, :]
    # mask_crop = mask[y:y+h, x:x+w]
    # return mask_crop, img_crop

    return mask, img


def get_mask_bbox(mask):
    """
    获取mask中所有mask的最小外接矩形，所以如果图中多个对象分布的范围很广，那就不太好弄
    """
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbox_list = []
    for ct in contours:
        if ct[2][0][0] == mask.shape[1] - 1:
            continue
        x, y, w, h = cv2.boundingRect(ct)
        bbox_list.append([x, y, x + w, y + h])

    bbox_list = np.array(bbox_list, dtype=np.int)
    xmin, xmax, ymin, ymax = bbox_list[:, 0].min(), bbox_list[:, 1].min(), bbox_list[:, 2].max(), bbox_list[:, 3].max()

    return xmin, xmax, ymin, ymax


def process_mask(img_src, mask_src, main_shape):
    """
    把所有mask的外接矩形区域提取出来，然后贴到与main图一样大小的0矩阵
    Args:
        img_src:
        mask_src:
        main_shape:

    Returns:

    """
    # 提取出mask的最小外接矩形
    xmin, ymin, xmax, ymax = get_mask_bbox(mask_src)
    # 截取对应区域
    et_img = img_src[ymin:ymax, xmin:xmax, :]

    # 本处的mask默认为单通道
    et_mask = mask_src[ymin:ymax, xmin:xmax]

    # 进行缩放，主办方给的图应该是适当缩小，而不是放大
    ratio = np.random.uniform(0.7, 0.9, 1)[0]
    new_col = int(et_img.shape[1] * ratio)
    new_row = int(et_img.shape[0] * ratio)

    # 不能让要贴的这部分宽高大于main图
    if new_col > main_shape[1]:
        col_r = main_shape[1] / et_img.shape[1]
        if new_row > main_shape[0]:
            row_r = main_shape[0] / et_img.shape[0]
            new_col = int(et_img.shape[1] * min(row_r, col_r))
            new_row = int(et_img.shape[0] * min(row_r, col_r))

        else:
            new_col = int(et_img.shape[1] * col_r)
            new_row = int(et_img.shape[0] * col_r)

    elif new_row > main_shape[0]:
        row_r = main_shape[0] / et_img.shape[0]

        new_col = int(et_img.shape[1] * row_r)
        new_row = int(et_img.shape[0] * row_r)

    et_img = cv2.resize(et_img, (new_col, new_row))
    et_mask = cv2.resize(et_mask, (new_col, new_row))

    # 进行亮度和对比度修改
    import copy
    ii_et = copy.deepcopy(et_img)
    et_img = RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=7)(image=et_img)["image"]
    show_two_image(ii_et, et_img)

    # show_two_image(et_img,et_mask)
    # 缩放后的图贴到与main_shape相同的0矩阵
    img_zero = np.zeros(main_shape, dtype=np.uint8)
    mask_zero = np.zeros(main_shape[:2], dtype=np.uint8)

    p_col = int(np.random.uniform(0, main_shape[1] - new_col, 1)[0])
    p_row = int(np.random.uniform(0, main_shape[0] - new_row, 1)[0])

    img_zero[p_row:p_row + new_row, p_col:p_col + new_col, :] = et_img
    mask_zero[p_row:p_row + new_row, p_col:p_col + new_col] = et_mask
    # show_two_image(img_zero,mask_zero)

    return img_zero, mask_zero


def copy_paste(mask_src, img_src, mask_main, img_main):
    rotate_list = []
    out_list = []

    #
    # mask_src_right = HorizontalFlip(p=1)(image=mask_src)
    # img_src_right= HorizontalFlip(p=1)(image=img_src)

    for degree in np.random.uniform(0, 360, 1):
        # for degree in np.random.uniform(0,360,36):

        _mask = Rotate(p=1, limit=(degree, degree))(image=mask_src)["image"]
        _img = Rotate(p=1, limit=(degree, degree))(image=img_src)["image"]
        rotate_list.append([_mask, _img])
        # show_two_image(mask_src,_mask)
        # show_two_image(img_src,_img)

    mask_main, img_main = random_flip_horizontal(mask_main, img_main)

    # rescale mask_src/img_src to less than mask_main/img_main's size
    h, w, _ = img_main.shape
    # mask_main=np.expand_dims(mask_main,axis=2)
    #
    # mask_main=np.concatenate((mask_main,mask_main,mask_main),axis=2)

    for i in range(len(rotate_list)):
        _mask, _img = rotate_list[i]
        # 处理src
        _img, _mask = process_mask(_img, _mask, img_main.shape)

        # _mask=np.expand_dims(_mask, axis=2)
        # _mask=np.concatenate((_mask,_mask,_mask),axis=2)
        # _mask, _img = rescale_src(_mask, _img, h, w)  # 如果是只使用rescale，那得到的mask和img在shape上与main的是一样的

        img = img_add(_img, img_main, _mask)
        mask = img_add(_mask, mask_main, _mask)

        # show_two_image(img_main,img)
        # show_two_image(mask_main,mask)
        out_list.append((img, mask))
    return out_list


def main(args):
    # background image path
    # JPEGs_main = os.path.join(args.main_dir, 'images',"val")
    JPEGs_main = args.main_dir

    segclass_src = os.path.join(args.src_dir, 'SegmentationClass')
    JPEGs_src = os.path.join(args.src_dir, 'JPEGImages')

    # create output path
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'JPEGImages'), exist_ok=True)

    masks_path_src = os.listdir(segclass_src)
    imgs_path_main = os.listdir(JPEGs_main)
    imgs_path_main = list(filter(lambda x: x.endswith(".tif"), imgs_path_main))
    for mask_path in tqdm.tqdm(masks_path_src):
        # get source mask and img
        mask_src = np.asarray(Image.open(os.path.join(segclass_src, mask_path)), dtype=np.uint8)
        img_src = cv2.imdecode(np.fromfile(os.path.join(JPEGs_src, mask_path.replace('.png', '.jpg')), dtype=np.uint8),
                               flags=1)

        zero_size = int(max(img_src.shape) * 1.5)  # 最长边的1.5倍，那样就能360度旋转不丢像素
        zero_mask = np.zeros((zero_size, zero_size), dtype=np.uint8)
        zero_img = np.zeros((zero_size, zero_size, img_src.shape[2]), dtype=np.uint8)

        mask_st_row = int(zero_mask.shape[0] // 2 - img_src.shape[0] // 2)
        mask_st_col = int(zero_mask.shape[1] // 2 - img_src.shape[1] // 2)
        img_st_row = int(zero_img.shape[0] // 2 - img_src.shape[0] // 2)
        img_st_col = int(zero_img.shape[1] // 2 - img_src.shape[1] // 2)

        # 原图放在中心位置，这样旋转的时候不会把mask对应部分像素丢掉
        zero_mask[mask_st_row:mask_st_row + mask_src.shape[0], mask_st_col:mask_st_col + mask_src.shape[1]] = mask_src
        zero_img[img_st_row:img_st_row + img_src.shape[0], img_st_col:img_st_col + img_src.shape[1], :] = img_src
        # show_two_image(img_src,zero_img)
        # show_two_image(mask_src,zero_mask)

        # random choice main mask/img
        img_main_path = np.random.choice(imgs_path_main)
        img_main = cv2.imdecode(
            np.fromfile(os.path.join(JPEGs_main, img_main_path), dtype=np.uint8), flags=1)
        # import copy
        # _ii=copy.deepcopy(img_main)

        img_size = 1200
        try:
            rs_ratio = img_size / min(img_main.shape[:2]) + 0.001
        except:
            pass
        img_main = cv2.resize(img_main, (int(img_main.shape[1] * rs_ratio), int(img_main.shape[0] * rs_ratio)))

        mask_main = np.zeros(img_main.shape[:2], dtype=np.uint8)
        # show_two_image(_ii,img_main)

        # Copy-Paste data augmentation
        out_list = copy_paste(zero_mask, zero_img, mask_main, img_main)

        for i in range(len(out_list)):
            img, mask = out_list[i]

            out_mask_dir = os.path.join(args.output_dir, 'SegmentationClass')
            mask_filename = "%d.png" % len(os.listdir(out_mask_dir))
            img_filename = mask_filename.replace('.png', '.jpg')
            save_colored_mask(mask, os.path.join(out_mask_dir, mask_filename))

            suffix = ".%s" % img_main_path.split('.')[-1]
            cv2.imencode(suffix, img)[1].tofile(os.path.join(args.output_dir, 'JPEGImages', img_filename))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", default=r"G:\背景军港\export_bg1", type=str,
                        help="to be pasted directory")
    parser.add_argument("--src_dir", default=r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\outship_cls_num\big_41",
                        type=str,
                        help="to be copyed directory")
    parser.add_argument("--output_dir", default=r"G:\hrsc\out_paste", type=str,
                        help="output dataset directory")
    parser.add_argument("--lsj", default=False, type=bool, help="if use Large Scale Jittering, now not using")
    return parser.parse_args()


if __name__ == '__main__':
    # "每个对象的mask用不同的像素值表示，这样方便把像素都叠加到一起后提取出单个对象的mask"
    args = get_args()
    main(args)
