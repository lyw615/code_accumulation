import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.visualize.visual_base import show_two_image

import pandas as pd
import os.path as osp

from PIL import Image, ImageDraw

csv_path = r"/home/lyw/t_ensemble.csv"
with open(csv_path, 'r') as f:
    record = f.readlines()
record.pop(0)
image_dir = r"/home/data1/yw/mmdetection/data/water_detection/split_folds/fold_v1/foldv1_yolo/test/tt/images"
for _ in record:
    _ = _.strip('\n').split(',')
    _[3:] = list(map(int, _[3:]))
    show_img = Image.open(osp.join(image_dir, _[1] + ".jpg"))
    draw = ImageDraw.Draw(show_img)
    draw.rectangle([(_[3], _[4]), (_[5], _[6])], width=3)
    import numpy as np

    mat_show = np.array(show_img)
    print(_[3:])
    show_two_image(mat_show, mat_show, _[1])

    print("pp")
