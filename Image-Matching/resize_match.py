import cv2, time, rasterio, os
import numpy as np
from matplotlib import pyplot as plt
from rasterio import Affine
from rasterio.enums import Resampling

"""
resize 50分之一后，大概1秒一张
对两个文件夹内的基准影像和测试影像遍历进行粗匹配，使用多进程加速
"""


def resample_raster(raster, scale=0.05):
    t = raster.transform

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = int(raster.height * scale)
    width = int(raster.width * scale)

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = raster.read(  # Note changed order of indexes, arrays are band, row, col order not row, col, band
        out_shape=(raster.count, height, width),
        resampling=Resampling.bilinear,
    )

    data = np.transpose(data, axes=(1, 2, 0))

    return data


def det_comp(img_path, scale):
    "得到给定路径图片的关键点和描述子"
    with  rasterio.open(img_path) as ds_b:
        raw_shape = (ds_b.height, ds_b.width)
        base_img = resample_raster(ds_b, scale=1 / scale)
        base_img_r = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = sift.detectAndCompute(base_img_r, None)

    return keypoints, descriptors, raw_shape, base_img_r


# test_img_path = r"E:\image_matching\test\布莱尔军港lit.tif"  # 这张图的driver可能不是GTIFF，导致速度很慢，所以这张图也是可以用rasterio缩放的
# base_img_path = r"E:\image_matching\base\brier_port.tif"

# test_img_dir = r"D:\BaiduNetdiskDownload\基准影像&测试影像示例\测试影像"
# base_img_dir = r"D:\BaiduNetdiskDownload\基准影像&测试影像示例\基准影像"

test_img_dir = r"E:\image_matching\test"
base_img_dir = r"E:\image_matching\base"

test_imgs = os.listdir(test_img_dir)
test_imgs = list(filter(lambda x: x.endswith('tif'), test_imgs))

base_imgs = os.listdir(base_img_dir)
base_imgs = list(filter(lambda x: x.endswith('tif'), base_imgs))

out_dir = r"E:\image_matching\out_dir"
out_txt_dir = os.path.join(out_dir, 'txt')
out_plot_dir = os.path.join(out_dir, 'match_plot')

os.makedirs(out_txt_dir, exist_ok=True)
os.makedirs(out_plot_dir, exist_ok=True)

st = time.time()
scale = 50  # 原图缩放为原来的1/50

# sift
sift = cv2.SIFT_create()
# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

for name_t in test_imgs:
    test_img_path = os.path.join(test_img_dir, name_t)

    test_img = cv2.imdecode(np.fromfile(test_img_path, dtype=np.uint8), 0)  # todo 有的图用rasterio打开没有cv快，但大部分都比cv快
    raw_shape_t = test_img.shape
    test_img_r = cv2.resize(test_img, (test_img.shape[0] // scale, test_img.shape[1] // scale))
    # test_img_r = cv2.cvtColor(test_img_r, cv2.COLOR_BGR2GRAY)

    keypoints_t, descriptors_t = sift.detectAndCompute(test_img_r, None)

    # keypoints_t, descriptors_t, raw_shape_t, test_img_r=det_comp(test_img_path,scale)

    min_points_dist = np.inf
    matched_base = None

    for name_b in base_imgs:
        base_img_path = os.path.join(base_img_dir, name_b)

        keypoints_b, descriptors_b, raw_shape_b, _ = det_comp(base_img_path, scale)
        # feature matching
        matches = bf.match(descriptors_b, descriptors_t)
        matches = sorted(matches, key=lambda x: x.distance)

        if matches[0].distance == 0:  # 如果出现关键点距离为0 ，很大概率匹配正确
            matched_base = name_b
            break

        points_dist = 0
        # 如果没有出现距离为0的关键点，那只能遍历完所有图片才能得到匹配结果
        for p in matches[:10]:  # 计算距离最短的10个关键点的距离
            points_dist += p.distance

        if min_points_dist > points_dist:
            min_points_dist = points_dist
            matched_base = name_b

    if len(base_imgs) > 0:
        base_imgs.remove(matched_base)
    else:
        break

    # 再匹配一次，获取对应的匹配结果
    keypoints_b, descriptors_b, raw_shape_b, base_img_r = det_comp(os.path.join(base_img_dir, matched_base), scale)

    # feature matching
    matches = bf.match(descriptors_b, descriptors_t)
    matches = sorted(matches, key=lambda x: x.distance)

    # queryIdx代表的特征点序列是keypoints_b中的，trainIdx代表的特征点序列是keypoints_t中的，此时这两张图中的特征点相互匹配,
    # 粗匹配由于缩放，会使得两个匹配点间的distance变大
    img3 = cv2.drawMatches(base_img_r, keypoints_b, test_img_r, keypoints_t, matches[:20], test_img_r,
                           flags=2)

    # 得到距离最小的匹配点
    nearest_point = matches[0]
    qdx = nearest_point.queryIdx
    tdx = nearest_point.trainIdx
    point_b = keypoints_b[qdx].pt  # (x,y),  up left as raw point
    point_t = keypoints_t[tdx].pt

    # 以此最近点为中心，选取测试影像里的部分区域进行精匹配
    tile_size_t = 1200
    # 关键点在原图上大概的像素坐标
    rt_x = raw_shape_t[1] * point_t[0] / test_img_r.shape[1]
    rt_y = raw_shape_t[0] * point_t[1] / test_img_r.shape[0]
    xmin_t = max(int(rt_x - tile_size_t / 2), 0)
    ymin_t = max(int(rt_y - tile_size_t / 2), 0)
    xmax_t = min(xmin_t + tile_size_t, raw_shape_t[1])
    ymax_t = min(ymin_t + tile_size_t, raw_shape_t[0])

    rt_rect = [xmin_t, ymin_t, xmax_t, ymax_t]  # 以关键点为中心取边长为tile_size的矩形

    # 基础影像也可以选择对应的中心点进行精匹配
    tile_size_b = tile_size_t
    rb_x = raw_shape_b[1] * point_b[0] / base_img_r.shape[1]
    rb_y = raw_shape_b[0] * point_b[1] / base_img_r.shape[0]
    xmin_b = max(int(rb_x - tile_size_b / 2), 0)
    ymin_b = max(int(rb_y - tile_size_b / 2), 0)
    xmax_b = min(xmin_b + tile_size_b, raw_shape_b[1])
    ymax_b = min(ymin_b + tile_size_b, raw_shape_b[0])
    rb_rect = [xmin_b, ymin_b, xmax_b, ymax_b]

    txt_name = "%s2%s.txt" % (matched_base.split('.')[0], name_t.split('.')[0])

    png_name = "%s2%s.png" % (matched_base.split('.')[0], name_t.split('.')[0])

    with open(os.path.join(out_txt_dir, txt_name), 'w', encoding='utf-8') as txt:  # 写入txt文件
        txt.write("%d %d %d %d \n" % (rt_rect[0], rt_rect[1], rt_rect[2], rt_rect[3]))  # 写入测试图片的
        txt.write("%d %d %d %d \n" % (rb_rect[0], rb_rect[1], rb_rect[2], rb_rect[3]))

    plt.imshow(img3)
    # plt.show()
    save_path = os.path.join(out_plot_dir, png_name)
    plt.savefig(save_path)

end = time.time()
print("time cost is  {}".format(end - st))
