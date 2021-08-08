import json, sys
import numpy as np
import os, rasterio
from rasterio.windows import Window
from tqdm import tqdm
import mmcv
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

sys.path.append("/home/data1/yw/github_projects/personal_github")
from code_aculat.visualize.visual_base import draw_multi_bboxes

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']
from mmdet.core.visualization import imshow_det_bboxes


def bbox_iou(box1, box2):
    """
    Calculate the IOU between box1 and box2.

    :param boxes: 2-d array, shape(n, 4)
    :param anchors: 2-d array, shape(k, 4)
    :return: 2-d array, shape(n, k)
    """
    # Calculate the intersection,
    # the new dimension are added to construct shape (n, 1) and shape (1, k),
    # so we can get (n, k) shape result by numpy broadcast
    box1 = box1[:, np.newaxis]  # [n, 1, 4]
    box2 = box2[np.newaxis]  # [1, k, 4]

    xx1 = np.maximum(box1[:, :, 0], box2[:, :, 0])
    yy1 = np.maximum(box1[:, :, 1], box2[:, :, 1])
    xx2 = np.minimum(box1[:, :, 2], box2[:, :, 2])
    yy2 = np.minimum(box1[:, :, 3], box2[:, :, 3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    area1 = (box1[:, :, 2] - box1[:, :, 0] + 1) * (box1[:, :, 3] - box1[:, :, 1] + 1)
    area2 = (box2[:, :, 2] - box2[:, :, 0] + 1) * (box2[:, :, 3] - box2[:, :, 1] + 1)
    ious = inter / (area1 + area2 - inter)

    return ious


label_ids = {name: i + 1 for i, name in enumerate(underwater_classes)}


def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def generate_json(img_root, annos, out_file):
    images = []
    annotations = []

    img_id = 1
    anno_id = 1
    for anno in tqdm(annos):
        img_name = anno[0]['image']
        img_path = os.path.join(img_root, img_name)
        w, h = Image.open(img_path).size
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)

        annotation = []
        for img_anno in anno:
            category_id = img_anno['category_id']
            xmin, ymin, w, h = img_anno['bbox']
            area = w * h
            segmentation = get_segmentation([xmin, ymin, w, h])
            annotation.append({
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [xmin, ymin, w, h],
                "category_id": category_id,
                "id": anno_id,
                "ignore": 0})
            anno_id += 1
        annotations.extend(annotation)
        img_id += 1
    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)


if __name__ == '__main__':
    np.random.seed(121)
    data_json_raw = json.load(
        open("/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/train/voc2coco.json", "r"))  # gt box
    data_json_pre = json.load(
        open("/home/data1/yw/data/iobjectspy_out/mmdetection/history_test_result/xf_result/xf_600_300_pre.bbox.json",
             "r"))  # pred box
    test_json = json.load(
        open("/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/train/test.json", "r"))  # pred box
    img = data_json_raw['images']

    image_dir = "/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold/fold_v4/Images"
    unclear_anno_img = []  # 看不清的图片，自己记录

    images = []
    gt_imgid2anno = {}  # 真实图像的box
    pred_imgid2anno = {}  # 预测图像的box
    imgid2name_gt = {}  # 图像名

    image_name_pre_none = []

    out_plot_dir = r"./gt_pre_plot_dir"
    os.makedirs(out_plot_dir, exist_ok=True)

    for imageinfo in data_json_raw['images']:  # 真实标注的image name
        imgid = imageinfo['id']
        imgid2name_gt[imgid] = imageinfo['file_name']  # get dict image id correspond to file name
    # print(len(imgid2name))  # 7600

    # gt image id 对pre image id 的映射
    imgid2name_pre = {}
    gt_id2pre_id = {}

    for imageinfo in test_json['images']:
        imgid = imageinfo['id']
        imgid2name_pre[imgid] = imageinfo['file_name']  # get dict image id correspond to file name

    # 创建映射字典
    _rev_dict_gt = dict(zip(imgid2name_gt.values(), imgid2name_gt.keys()))
    _rev_dict_pre = dict(zip(imgid2name_pre.values(), imgid2name_pre.keys()))

    for _name in _rev_dict_gt:
        gt_id2pre_id[_rev_dict_gt[_name]] = _rev_dict_pre[_name]

    for anno in data_json_raw['annotations']:  # 真实标签
        img_id = anno['image_id']
        if img_id not in gt_imgid2anno:
            gt_imgid2anno[img_id] = []
        gt_imgid2anno[img_id].append(anno)

    for anno in data_json_pre:  # 预测标签
        img_id = anno['image_id']
        if img_id not in pred_imgid2anno:
            pred_imgid2anno[img_id] = []
        pred_imgid2anno[img_id].append(anno)

    revised_annos = []
    num_iou_little = 0
    for imgid in tqdm(gt_imgid2anno.keys()):  # 一张张的比较
        if imgid2name_gt[imgid] in unclear_anno_img:  # 看不清的图像，不加入训练
            # print(imgid2name[imgid])
            continue
        pre_id = gt_id2pre_id[imgid]

        if pre_id not in pred_imgid2anno.keys():  # gt中没有被模型预测出有目标的图片
            image_name_pre_none.append(imgid2name_pre[pre_id])
            continue

        annos = pred_imgid2anno[pre_id]
        pred_boxes = []
        for anno in annos:
            xmin, ymin, w, h = anno['bbox']
            xmax = xmin + w
            ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = anno['score']
            class_id = int(anno['category_id']) - 1
            pred_boxes.append([xmin, ymin, xmax, ymax, confidence, class_id])
        pred_boxes = np.array(pred_boxes)
        pred_boxes = pred_boxes[pred_boxes[:, 4] > 0.4]  # 过滤掉低score

        gt_boxes = []
        revised_gt = []
        if imgid in gt_imgid2anno.keys():
            for anno in gt_imgid2anno[imgid]:
                xmin, ymin, w, h = anno['bbox']
                xmax = xmin + w
                ymax = ymin + h
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                class_id = int(anno['category_id']) - 1
                gt_boxes.append([xmin, ymin, xmax, ymax, class_id])
            gt_boxes = np.array(gt_boxes)

        if len(pred_boxes) == 0:  # 当前img没有预测框
            # 不用修正gt box的类别,但需把gt内的标注补充进去
            for anno in gt_imgid2anno[imgid]:
                revised_gt.append({'image': imgid2name_gt[imgid],
                                   'bbox': anno['bbox'], 'category_id': anno['category_id']})
            revised_annos.append(revised_gt)
            # filename = os.path.join('../underwater_data/train/image', imgid2name[imgid])
            # img = cv2.imread(filename)
            # basename = os.path.basename(filename)
            # imshow_det_bboxes(img, gt_boxes[:, :4], gt_boxes[:, 4], class_names=underwater_classes,
            #                   show=False,
            #                   out_file=os.path.join('../underwater_data/train/no_pred_box-0.95/' + basename))
            continue

        ious = bbox_iou(pred_boxes[:, :4], gt_boxes[:, :4])  # [n, k]  如果都只有一个,还小于0.75,那需要可视化排查,是预测错还是gt有问题

        # read tif
        img_path = os.path.join(image_dir, imgid2name_gt[imgid])
        with rasterio.open(img_path) as ds:

            for _ in range(len(ious)):
                iou_per_box = ious[_]

                max_idx = np.argmax(iou_per_box, axis=0)  # [k,]
                max_value = np.amax(iou_per_box, axis=0)  # [k,]

                if max_value > 0.75:
                    continue

                box1_pre = pred_boxes[_][:4]  # [:4] is box
                box2_gt = gt_boxes[max_idx][:4]

                # #iou decide region
                if max_value > 0.1:
                    xmin = min(box1_pre[0], box2_gt[0])
                    ymin = min(box1_pre[1], box2_gt[1])
                    xmax = max(box1_pre[2], box2_gt[2])
                    ymax = max(box1_pre[3], box2_gt[3])

                    xmin = int(max(0, xmin - 50))
                    ymin = int(max(0, ymin - 50))
                    xmax = int(min(ds.width, xmax + 50))
                    ymax = int(min(ds.height, ymax + 50))

                    # get region to array
                    block = ds.read(window=Window(xmin, ymin, xmax - xmin, ymax - ymin))
                    block = block[:3, :, :]

                    box1_pre[[0, 2]] -= xmin
                    box1_pre[[1, 3]] -= ymin

                    box2_gt[[0, 2]] -= xmin
                    box2_gt[[1, 3]] -= ymin

                    # array to image
                    block = np.transpose(block, axes=(1, 2, 0))
                    # image draw
                    convert_img = Image.fromarray(block)
                    draw_multi_bboxes(convert_img, [box1_pre, box2_gt], color=['blue', 'red'])  # draw multi box

                    plt.imshow(convert_img)
                    # plt.show()
                    save_name = "%s_%.4f.png" % (os.path.basename(img_path).split('.')[0], max_value)
                    save_path = os.path.join(out_plot_dir, save_name)
                    plt.savefig(save_path)
                    # convert_img.show()
                else:
                    num_iou_little += 1

                    # for box in [box1_pre,box2_gt]:    #如果两个box的iou过低,也可以分别把box可视化
                    #     xmin,ymin,xmax,ymax=box
                    #     xmin = int(max(0, xmin - 50))
                    #     ymin = int(max(0, ymin - 50))
                    #     xmax = int(min(ds.width, xmax + 50))
                    #     ymax = int(min(ds.height, ymax + 50))
                    #
                    #     # get region to array
                    #     block = ds.read(window=Window(xmin, ymin, xmax - xmin, ymax - ymin))
                    #     block = block[:3, :, :]
                    #
                    #     box[ [0, 2]] -= xmin
                    #     box[[1, 3]] -= ymin
                    #
                    #
                    #     # array to image
                    #     block = np.transpose(block, axes=(1, 2, 0))
                    #     # image draw
                    #     convert_img = Image.fromarray(block)
                    #     draw_multi_bboxes(convert_img, [box])  # draw multi box
                    #
                    #     plt.imshow(convert_img)
                    #     plt.show()
                    #     # convert_img.show()

    print("iou lower than 0.1 numbers  %d" % num_iou_little)

    if len(image_name_pre_none) > 0:
        with open('./image_pre_none.txt', 'w', encoding='utf-8') as txt:
            txt.writelines(image_name_pre_none)
