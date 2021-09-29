import os, platform, shutil
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv


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


def draw_bboxes_on_image(image_path, bboxes, class_names, title=None, save_dir=None, save_name=None):
    """
    bbox:[[xmin,ymin,xmax,ymax]] list
    """
    # class_id2name = {10:"航母",11:"两栖",12:"驱护",13:"导弹巡逻艇",14:"扫雷舰艇",15:"登陆",16:"补给",17:"支援",18:"濒海",19:"侦察"}
    class_id2name = get_zw_from_txt("/home/data1/yw/github_projects/personal_github/code_aculat/zw.txt")
    if isinstance(image_path, str):
        try:
            show_img = Image.open(image_path)
            title = os.path.basename(image_path)
        except:
            show_img = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
            show_img = Image.fromarray(show_img)

    else:
        show_img = Image.fromarray(image_path)

    draw = ImageDraw.Draw(show_img)

    for _ in range(len(bboxes)):
        bbox = bboxes[_]
        class_name = class_names[_]

        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], width=4,
                       outline='red')  # (xmin,ymin) ,(xmax,ymax)

        if platform.system() == "Linux":
            txt_font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"  # 英文字体
            # txt_font = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"   #中文字体
            txt_font = ImageFont.truetype(txt_font, size=30)
            draw.text((int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)), '%s' % class_id2name[class_name],
                      font=txt_font)
        else:
            draw.text((int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)), '%s' % class_name)

    # show_img.save(os.path.join(save_dir,save_name))
    show_one_image(np.array(show_img), title, save_dir=save_dir, save_name=save_name)


def draw_bbox(image, bbox):
    """
    Draw one bounding box on image.
    Args:
        image (PIL.Image): a PIL Image object.
        bbox (np.array|list|tuple): (xmin, ymin, xmax, ymax).
    """
    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = bbox
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top),
         (left, top)],
        width=4,
        fill='red')


def draw_multi_bboxes(image, bboxes, color=None):
    """
    Draw multi bounding box on image.
    Args:
        image (PIL.Image): a PIL Image object.
        bbox (np.array|list|tuple): (xmin, ymin, xmax, ymax).
    """
    if not color:
        color = ['red'] * len(bboxes)

    for _ in range(len(bboxes)):
        bbox = bboxes[_]
        draw = ImageDraw.Draw(image)
        xmin, ymin, xmax, ymax = bbox
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=4,
            fill=color[_])


def draw_multi_bboxes_with_names(image, bboxes, class_names, color=None):
    """
    Draw multi bounding box on image.
    Args:
        image (PIL.Image): a PIL Image object.
        bbox (np.array|list|tuple): (xmin, ymin, xmax, ymax).
    """
    if not color:
        color = ['red'] * len(bboxes)

    for _ in range(len(bboxes)):
        bbox = bboxes[_]
        draw = ImageDraw.Draw(image)
        xmin, ymin, xmax, ymax = bbox
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=4,
            fill=color[_])

        class_name = class_names[_]

        if platform.system() == "Linux":
            txt_font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            txt_font = ImageFont.truetype(txt_font, size=70)
            draw.text((int((bbox[0] + bbox[2]) / 2), bbox[3]), '%s' % class_name, font=txt_font)
        else:
            draw.text((int((bbox[0] + bbox[2]) / 2), bbox[3]), '%s' % class_name)


def draw_multi_rotated_bboxes_with_names(image, bboxes, class_names, color=None):
    """
    Draw multi bounding box on image.
    Args:
        image (PIL.Image): a PIL Image object.
        bbox (np.array|list|tuple): (xmin, ymin, xmax, ymax).
    """
    if not color:
        color = ['red'] * len(bboxes)
    bboxes = np.array(bboxes, dtype=np.int)
    for _ in range(len(bboxes)):
        bbox = bboxes[_]
        draw = ImageDraw.Draw(image)
        bbox0, bbox1, bbox2, bbox3 = bbox

        draw.line(
            [(bbox0[0], bbox0[1]), (bbox1[0], bbox1[1]), (bbox2[0], bbox2[1]), (bbox3[0], bbox3[1]),
             (bbox0[0], bbox0[1])],
            width=4,
            fill=color[_])

        class_name = class_names[_]

        if platform.system() == "Linux":
            txt_font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            # txt_font = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # 中文字体
            txt_font = ImageFont.truetype(txt_font, size=70)
            draw.text((int((bbox0[0] + bbox2[0]) / 2), int((bbox1[1] + bbox3[1]) / 2)), '%s' % class_name,
                      font=txt_font)
        else:
            draw.text((int((bbox[0] + bbox[2]) / 2), bbox[3]), '%s' % class_name)


def show_bboxes_region(image_path, bboxes, class_names, out_dir, title=None):
    """
    bbox:[[xmin,ymin,xmax,ymax]] list
    主要针对大的tif影像,此时仅将bbox周围区域可视化,每个bbox会存成一张图
    """
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(image_path) as ds:
        for _ in range(len(bboxes)):
            box = bboxes[_]
            box = np.array(box, dtype=np.int)
            class_name = class_names[_]
            xmin, ymin, xmax, ymax = box

            if xmin >= xmax or ymin >= ymax:
                # print("error anno tif is  {}".format(os.path.basename(image_path)))
                shutil.copy(image_path, os.path.join(out_dir, os.path.basename(image_path)))
                continue

            xmin = int(max(0, xmin - 50))
            ymin = int(max(0, ymin - 50))
            xmax = int(min(ds.width, xmax + 50))
            ymax = int(min(ds.height, ymax + 50))

            # get region to array
            block = ds.read(window=Window(xmin, ymin, xmax - xmin, ymax - ymin))
            block = block[:3, :, :]

            box[[0, 2]] -= xmin
            box[[1, 3]] -= ymin

            # array to image
            block = np.transpose(block, axes=(1, 2, 0))
            # image draw
            convert_img = Image.fromarray(block)
            draw_multi_bboxes_with_names(convert_img, [box], class_name)  # draw multi box or one box

            plt.imshow(convert_img)
            # plt.show()
            save_name = "%s_%s_%d.png" % (os.path.basename(image_path).split('.')[0], class_name, _)
            save_path = os.path.join(out_dir, save_name)
            plt.savefig(save_path)


def get_zw_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as ff:
        record = ff.readlines()[0]

    record = record.split(",")
    key_ind = 10
    zw_dict = {}
    for i in record:
        zw_dict[key_ind] = i
        key_ind += 1
    return zw_dict
