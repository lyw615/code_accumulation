# -*- coding: utf-8 -*-
"""
@Author  : zengwb
@Time    : 2021/4/17
@Software: PyCharm
"""
import copy, sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..", "..")))

import cv2, time, codecs
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
from tqdm import trange  # 显示进度条
from multiprocessing import cpu_count  # 查看cpu核心数
from multiprocessing import Pool
from code_aculat.visualize.visual_base import draw_bboxes_on_image


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def deal_xml(xml_f, category_dict):
    tree = ET.parse(xml_f)
    root = tree.getroot()
    object_list = []

    # 处理每个标注的检测框
    for obj in get(root, 'object'):
        # 取出检测框类别名称
        category = get_and_check(obj, 'name', 1).text
        if category not in category_dict.keys():
            category_dict[category] = len(category_dict.keys()) + 1

        # 更新类别ID字典
        bndbox = get_and_check(obj, 'bndbox', 1)
        xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
        ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
        xmax = int(get_and_check(bndbox, 'xmax', 1).text)
        ymax = int(get_and_check(bndbox, 'ymax', 1).text)
        assert (xmax > xmin)
        assert (ymax > ymin)

        obj_info = [xmin, ymin, xmax, ymax, category_dict[category]]
        object_list.append(obj_info)

    return object_list, category_dict


def exist_objs(list_1, list_2, sliceHeight, sliceWidth, min_h=15, min_w=15):
    '''
    list_1:当前slice的图像
    list_2:原图中的所有目标
    min_h,min_w 有些目标GT会被窗口切分，太小的丢掉，即这两个变量为目标对象的最小宽高
    return:原图中位于当前slicze中的目标集合
    '''
    return_objs = []
    s_xmin, s_ymin, s_xmax, s_ymax = list_1[0], list_1[1], list_1[2], list_1[3]
    for vv in list_2:
        xmin, ymin, xmax, ymax, category = vv[0], vv[1], vv[2], vv[3], vv[4]
        # 1111111
        if s_xmin <= xmin < s_xmax and s_ymin <= ymin < s_ymax:  # 目标点的左上角在切图区域中
            if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                x_new = xmin - s_xmin
                y_new = ymin - s_ymin
                return_objs.append([x_new, y_new, x_new + (xmax - xmin), y_new + (ymax - ymin), category])
        if s_xmin <= xmin < s_xmax and ymin < s_ymin:  # 目标点的左上角在切图区域上方
            # 22222222
            if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                x_new = xmin - s_xmin
                y_new = 0
                if xmax - s_ymax - x_new > min_w and ymax - s_ymax - y_new > min_h:
                    return_objs.append([x_new, y_new, xmax - s_ymax, ymax - s_ymax, category])
            # 33333333
            if xmax > s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域右方
                x_new = xmin - s_xmin
                y_new = 0
                if s_xmax - s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                    return_objs.append([x_new, y_new, s_xmax - s_xmin, ymax - s_ymin, category])
        if s_ymin < ymin <= s_ymax and xmin < s_xmin:  # 目标点的左上角在切图区域左方
            # 444444
            if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                x_new = 0
                y_new = ymin - s_ymin
                if xmax - s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                    return_objs.append([x_new, y_new, xmax - s_xmin, ymax - s_ymin, category])
            # 555555
            if s_xmin < xmax < s_xmax and ymax >= s_ymax:  # 目标点的右下角在切图区域下方
                x_new = 0
                y_new = ymin - s_ymin
                if xmax - s_xmin - x_new > min_w and s_ymax - s_ymin - y_new > min_h:
                    return_objs.append([x_new, y_new, xmax - s_xmin, s_ymax - s_ymin, category])
        # 666666
        if s_xmin >= xmin and ymin <= s_ymin:  # 目标点的左上角在切图区域左上方
            if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                x_new = 0
                y_new = 0
                if xmax - s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                    return_objs.append([x_new, y_new, xmax - s_xmin, ymax - s_ymin, category])
        # 777777
        if s_xmin <= xmin < s_xmax and s_ymin <= ymin < s_ymax:  # 目标点的左上角在切图区域中
            if ymax >= s_ymax and xmax >= s_xmax:  # 目标点的右下角在切图区域右下方
                x_new = xmin - s_xmin
                y_new = ymin - s_ymin
                if s_xmax - s_xmin - x_new > min_w and s_ymax - s_ymin - y_new > min_h:
                    return_objs.append([x_new, y_new, s_xmax - s_xmin, s_ymax - s_ymin, category])
            # 8888888
            if s_xmin < xmax < s_xmax and ymax >= s_ymax:  # 目标点的右下角在切图区域下方
                x_new = xmin - s_xmin
                y_new = ymin - s_ymin
                if xmax - s_xmin - x_new > min_w and s_ymax - s_ymin - y_new > min_h:
                    return_objs.append([x_new, y_new, xmax - s_xmin, s_ymax - s_ymin, category])
            # 999999
            if xmax > s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域右方
                x_new = xmin - s_xmin
                y_new = ymin - s_ymin
                if s_xmax - s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                    return_objs.append([x_new, y_new, s_xmax - s_xmin, ymax - s_ymin, category])

    return return_objs


def crop_objs(crop_bounds, label_bboxes, area_thresh=0.7):
    """

    Args:
        crop_bounds:切出的tile在原图上的坐标区域 [xmin, ymin, xmax, ymax]
        label_bboxes:从xml里提取出的所有对象的坐标

    Returns:返回切出crop的tile里对象的[xmin, ymin, xmax, ymax, class_name]

    """

    x1_drop_index = np.where(label_bboxes[:, 0] > crop_bounds[2])[0]  # 右侧bbox
    x2_drop_index = np.where(label_bboxes[:, 2] < crop_bounds[0])[0]  # 左侧bbox

    y1_drop_index = np.where(label_bboxes[:, 1] > crop_bounds[3])[0]  # 下侧bbox
    y2_drop_index = np.where(label_bboxes[:, 3] < crop_bounds[1])[0]  # 上侧bbox

    drop_index = np.concatenate((x1_drop_index, x2_drop_index, y1_drop_index, y2_drop_index))
    drop_index = np.unique(drop_index)

    save_index = [x for x in range(len(label_bboxes)) if x not in drop_index]
    if len(save_index) == 0:
        return []

    region_boxes = label_bboxes[save_index]

    region_boxes[:, [0, 2]] = region_boxes[:, [0, 2]] - crop_bounds[0]
    region_boxes[:, [1, 3]] = region_boxes[:, [1, 3]] - crop_bounds[1]
    raw_region_boxes = copy.deepcopy(region_boxes)  # 用于计算crop后box的面积保留比例

    # 处理xmax,ymax都超出范围的
    region_boxes[:, 2] = np.where(region_boxes[:, 2] > crop_bounds[2], crop_bounds[2], region_boxes[:, 2])
    region_boxes[:, 3] = np.where(region_boxes[:, 3] > crop_bounds[3], crop_bounds[3], region_boxes[:, 3])

    # 计算保留的比例
    region_area = (region_boxes[:, 2] - region_boxes[:, 0]) * (region_boxes[:, 3] - region_boxes[:, 1])
    raw_area = (raw_region_boxes[:, 2] - raw_region_boxes[:, 0]) * (raw_region_boxes[:, 3] - raw_region_boxes[:, 1])

    saved_portion = region_area / raw_area
    area_filter = np.where(saved_portion > area_thresh)[0]  # 过滤掉面积保留比例低于这个值的
    saved_label = region_boxes[area_filter]

    return saved_label


def bbox_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s2  # + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou


def exist_objs_iou(list_1, list_2, sliceHeight, sliceWidth, win_h, win_w, iou_thresh=0.2):
    # 根据iou判断框是否保留，并返回bbox
    return_objs = []
    s_xmin, s_ymin, s_xmax, s_ymax = list_1[0], list_1[1], list_1[2], list_1[3]

    for single_box in list_2:
        xmin, ymin, xmax, ymax, category = single_box[0], single_box[1], single_box[2], single_box[3], single_box[4]
        iou = bbox_iou(list_1, single_box[:4])
        if iou > iou_thresh:
            if iou == 1:
                x_new = xmin - s_xmin
                y_new = ymin - s_ymin
                return_objs.append([x_new, y_new, x_new + (xmax - xmin), y_new + (ymax - ymin), category])
            else:
                xlist = np.sort([xmin, xmax, s_xmin, s_xmax])
                ylist = np.sort([ymin, ymax, s_ymin, s_ymax])
                # print(win_h, win_w, list_1, single_box, xlist[1] - s_xmin, ylist[1] - s_ymin)
                return_objs.append(
                    [xlist[1] - s_xmin, ylist[1] - s_ymin, xlist[2] - s_xmin, ylist[2] - s_ymin, category])
    return return_objs


def make_slice_voc(image_out_path, out_voc_dir, exiset_obj_list, categry_dict, sliceHeight=1024, sliceWidth=1024):
    name = image_out_path.split('\\')[-1]

    with codecs.open(os.path.join(out_voc_dir, name[:-4] + '.xml'), 'w', 'utf-8') as xml:
        xml.write('<annotation>\n')
        xml.write('\t<filename>' + name + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(sliceWidth) + '</width>\n')
        xml.write('\t\t<height>' + str(sliceHeight) + '</height>\n')
        xml.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml.write('\t</size>\n')
        cnt = 1
        for obj in exiset_obj_list:
            #
            bbox = obj[:4]
            class_name = categry_dict[obj[-1]]
            xmin, ymin, xmax, ymax = bbox

            xml.write('\t<object>\n')
            xml.write('\t\t<name>' + class_name + '</name>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
            cnt += 1
        assert cnt > 0
        xml.write('</annotation>')


###############################################################################
def slice_im(List_subsets, out_image_dir, out_voc_dir, raw_images_dir, raw_ann_dir, i=None, sliceHeight=640,
             sliceWidth=640, overlap=0.2, area_thresh=0.7):
    """

    Args:
        List_subsets:
        outdir:
        raw_images_dir:
        raw_ann_dir:
        i:
        sliceHeight: 切片的高
        zero_frac_thresh: 0像素所占比例的限制，超出这个比例将不会保存这张图
        overlap: 前后两张图的重叠比例，一张切片实际滑动的步长：  （1-overlap)*切片的宽或者高
                 如果是整数，那就是直接指定重叠像素值

    Returns:

    """
    cnt = 0
    categry_dict = {}
    for per_img_name in tqdm(List_subsets):

        o_name, _ = os.path.splitext(per_img_name)
        out_name = str(o_name) + '_' + str(cnt)
        image_path = os.path.join(raw_images_dir, per_img_name)
        ann_path = os.path.join(raw_ann_dir, per_img_name[:-4] + '.xml')

        image0 = cv2.imread(image_path, 1)  # color
        ext = '.' + image_path.split('.')[-1]
        win_h, win_w = image0.shape[:2]

        object_list, categry_dict = deal_xml(ann_path, categry_dict)
        object_array = np.array(object_list)

        if isinstance(overlap, float):

            dx = int((1. - overlap) * sliceWidth)  # 每次滑动，真实移动的像素
            dy = int((1. - overlap) * sliceHeight)

        else:
            dx = sliceWidth - overlap  # 如果是整数，那就是直接指定重叠像素值
            dy = sliceHeight - overlap

        for y0 in range(0, image0.shape[0], dy):  # 如果没有出现超出边界的情况，那么y0,x0是每个tile在原图上的左上角坐标，即y,x
            for x0 in range(0, image0.shape[1], dx):

                # 这一步确保了不会出现比要切的图像小的图，其实是通过调整最后的overlop来实现的
                # 举例:h=6000,w=8192.若使用640来切图,overlop:0.2*640=128,间隔就为512.所以小图的左上角坐标的纵坐标y0依次为:
                #:0,512,1024,....,5120,接下来并非为5632,因为5632+640>6000,所以y0=6000-640
                if y0 + sliceHeight > image0.shape[0]:
                    y = image0.shape[0] - sliceHeight
                else:
                    y = y0  # 如果y0+tile 的高没有超出图片，那么上次tile的边界就是y的值
                if x0 + sliceWidth > image0.shape[1]:
                    x = image0.shape[1] - sliceWidth
                else:
                    x = x0

                slice_xmax = x + sliceWidth  # 本次tile在原图上的边界坐标
                slice_ymax = y + sliceHeight

                # exiset_obj_list = exist_objs([x, y, slice_xmax, slice_ymax], object_list, sliceHeight, sliceWidth)
                exiset_obj_array = crop_objs([x, y, slice_xmax - 1, slice_ymax - 1], object_array, area_thresh)
                # exiset_obj_list = exist_objs_iou([x,y,slice_xmax,slice_ymax],object_list, sliceHeight, sliceWidth, win_h, win_w)
                if len(exiset_obj_array) > 0:  # 如果为空,说明切出来的这一张图不存在目标
                    # extract image
                    window_c = image0[y:y + sliceHeight, x:x + sliceWidth]

                    image_out_path = os.path.join(out_image_dir, out_name + \
                                                  '_' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(
                        sliceWidth) + '_' + str(win_w) + '_' + str(win_h) + ext)

                    cnt += 1

                    cv2.imwrite(image_out_path, window_c)
                    # ------制作新的xml------
                    categry_dict_rev = {}
                    for key, value in categry_dict.items():
                        categry_dict_rev[value] = key

                    make_slice_voc(image_out_path, out_voc_dir, exiset_obj_array, categry_dict_rev, sliceHeight,
                                   sliceWidth)

                    # draw_bboxes_on_image(window_c,exiset_obj_array[:,:4],exiset_obj_array[:,4])   #直接可视化切图后的标签叠加图


if __name__ == "__main__":
    not_use_multiprocessing = True
    raw_images_dir = r'E:\push_branch\resources_ml\out\training\object_ext_train_data\Images'  # 这里就是原始的图片
    # raw_images_dir = r'E:\push_branch\resources_ml\out\one_image'  # 这里就是原始的图片
    raw_ann_dir = r'E:\push_branch\resources_ml\out\training\object_ext_train_data\Annotations'

    output_dir = r"E:\push_branch\resources_ml\out\clip_xml"
    out_voc_dir = os.path.join(output_dir, "Annotations")  # 切出来的标签也保存为voc格式
    out_image_dir = os.path.join(output_dir, "Images")
    os.makedirs(out_voc_dir, exist_ok=True)
    os.makedirs(out_image_dir, exist_ok=True)

    sliceHeight = 128
    sliceWidth = 128
    overlap = 30    #最好设置为最大目标对象的1.2倍，被切断时至少能保留一边完整
    area_thresh = 0.7

    List_imgs = os.listdir(raw_images_dir)

    if not_use_multiprocessing:

        slice_im(List_imgs, out_image_dir, out_voc_dir, raw_images_dir, raw_ann_dir, sliceWidth=sliceWidth,
                 sliceHeight=sliceHeight,
                 overlap=overlap, area_thresh=area_thresh)  # 切片的宽高,切片的重叠，label的面积保留比例
    else:
        Len_imgs = len(List_imgs)  # 数据集长度
        num_cores = cpu_count()  # cpu核心数
        # print(num_cores, Len_imgs)
        if num_cores >= 8:  # 八核以上，将所有数据集分成八个子数据集
            num_cores = 8
            subset1 = List_imgs[:Len_imgs // 8]
            subset2 = List_imgs[Len_imgs // 8: Len_imgs // 4]
            subset3 = List_imgs[Len_imgs // 4: (Len_imgs * 3) // 8]
            subset4 = List_imgs[(Len_imgs * 3) // 8: Len_imgs // 2]
            subset5 = List_imgs[Len_imgs // 2: (Len_imgs * 5) // 8]
            subset6 = List_imgs[(Len_imgs * 5) // 8: (Len_imgs * 6) // 8]
            subset7 = List_imgs[(Len_imgs * 6) // 8: (Len_imgs * 7) // 8]
            subset8 = List_imgs[(Len_imgs * 7) // 8:]

            List_subsets = [subset1, subset2, subset3, subset4, subset5, subset6, subset7, subset8]

        p = Pool(num_cores)
        for i in range(num_cores):
            p.apply_async(slice_im, args=(
            List_subsets[i], out_image_dir, out_voc_dir, raw_images_dir, raw_ann_dir, i, sliceWidth, sliceHeight,
            overlap, area_thresh))
        p.close()
        p.join()
