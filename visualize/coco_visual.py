from pycocotools.coco import COCO
from tqdm import tqdm
import cv2, os
import numpy as np
from matplotlib import pyplot as plt

image_dir = r"/home/data1/yw/data/iobjectspy_out/problem_data"
anno_json_path = r"/home/data1/yw/data/iobjectspy_out/pred.json"

coco = COCO(anno_json_path)
imgIds = coco.getImgIds()
catIds = coco.getCatIds()

for imgId in tqdm(imgIds):
    img = coco.loadImgs(imgId)[0]
    file_name = img['file_name']
    image_arr = cv2.imdecode(np.fromfile(os.path.join(image_dir, file_name), dtype=np.uint8), flags=1)
    plt.imshow(image_arr)

    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=0)
    anns = coco.loadAnns(annIds)

    coco.showAnns(anns, draw_bbox=True)

    plt.show()
