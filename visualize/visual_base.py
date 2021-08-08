import os, platform

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


def show_one_image(image, title=None):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)

    if title:
        plt.title(title)
    plt.show()


def draw_bboxes_on_image(image_path, bboxes, class_names, title=None):
    """
    bbox:[[xmin,ymin,xmax,ymax]] list

    """

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
            txt_font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            txt_font = ImageFont.truetype(txt_font, size=70)
            draw.text((int((bbox[0] + bbox[2]) / 2), bbox[3]), '%s' % class_name, font=txt_font)
        else:
            draw.text((int((bbox[0] + bbox[2]) / 2), bbox[3]), '%s' % class_name)

    show_one_image(np.array(show_img), title)


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
    Draw one bounding box on image.
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
