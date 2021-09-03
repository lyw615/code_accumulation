from albumentations import Rotate, Blur, CLAHE, MotionBlur, GaussianBlur, RandomShadow, RandomBrightnessContrast, \
    Compose, OneOf
import os, cv2
from albumentations import HorizontalFlip, VerticalFlip
import numpy as np
from PIL import Image
import imgviz


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


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


mask_dir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\outship_cls_num\big_72\SegmentationClass"
img_dir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\outship_cls_num\big_72\images\train"

out_dir = os.path.join(os.path.dirname(img_dir), "out_dir")
out_mask_dir = os.path.join(out_dir, 'mask')
out_img_dir = os.path.join(out_dir, 'img')

os.makedirs(out_mask_dir, exist_ok=True)
os.makedirs(out_img_dir, exist_ok=True)

masks = os.listdir(mask_dir)

for mask in masks:
    img_name = mask.replace('.png', '.jpg')

    img_list = []
    mask_list = []

    mask_arr_raw = cv2.imdecode(np.fromfile(os.path.join(mask_dir, mask), dtype=np.uint8), 1)
    mask_arr = mask_arr_raw[:, :, 1]
    if mask_arr.max() < 1:
        mask_arr = mask_arr_raw[:, :, 2]
        if mask_arr.max < 1:
            raise ("max mask value low than 1")
    img_arr = cv2.imdecode(np.fromfile(os.path.join(img_dir, img_name), dtype=np.uint8), 1)  # imencode会自动把bgr转成rgb

    ts_img = np.transpose(img_arr, axes=(1, 0, 2))
    ts_mk = np.transpose(mask_arr, axes=(1, 0))

    # show_two_image(img_arr,ts_img)
    # out_img_name = "%s.jpg" % len(os.listdir(out_img_dir))
    # cv2.imencode('.jpg', ts_img)[1].tofile(os.path.join(out_img_dir, out_img_name))
    # out_mk_name=out_img_name.replace(".jpg",'.png')
    # save_colored_mask(ts_mk, os.path.join(out_mask_dir,out_mk_name))

    img_list.append(img_arr)
    mask_list.append(mask_arr)
    img_list.append(ts_img)
    mask_list.append(ts_mk)

    for num in range(len(img_list)):
        out_img_name = "%s.jpg" % len(os.listdir(out_img_dir))
        cv2.imencode('.jpg', img_list[num])[1].tofile(os.path.join(out_img_dir, out_img_name))
        out_mk_name = out_img_name.replace(".jpg", '.png')
        save_colored_mask(mask_list[num], os.path.join(out_mask_dir, out_mk_name))

        hf_img = HorizontalFlip().apply(img_list[num])
        hf_mk = HorizontalFlip().apply(mask_list[num])

        out_img_name = "%s.jpg" % len(os.listdir(out_img_dir))
        cv2.imencode('.jpg', hf_img)[1].tofile(os.path.join(out_img_dir, out_img_name))
        out_mk_name = out_img_name.replace(".jpg", '.png')
        save_colored_mask(hf_mk, os.path.join(out_mask_dir, out_mk_name))

        vf_img = VerticalFlip().apply(img_list[num])
        vf_mk = VerticalFlip().apply(mask_list[num])
        out_img_name = "%s.jpg" % len(os.listdir(out_img_dir))
        cv2.imencode('.jpg', vf_img)[1].tofile(os.path.join(out_img_dir, out_img_name))
        out_mk_name = out_img_name.replace(".jpg", '.png')
        save_colored_mask(vf_mk, os.path.join(out_mask_dir, out_mk_name))

        vh_img = HorizontalFlip().apply(vf_img)
        vh_mk = HorizontalFlip().apply(vf_mk)
        out_img_name = "%s.jpg" % len(os.listdir(out_img_dir))
        cv2.imencode('.jpg', vh_img)[1].tofile(os.path.join(out_img_dir, out_img_name))
        out_mk_name = out_img_name.replace(".jpg", '.png')
        save_colored_mask(vh_mk, os.path.join(out_mask_dir, out_mk_name))
