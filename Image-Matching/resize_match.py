import cv2, time, rasterio
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


test_img_path = r"E:\image_matching\test\布莱尔军港lit.tif"  # 这张图的driver可能不是GTIFF，导致速度很慢，所以这张图也是可以用rasterio缩放的
base_img_path = r"E:\image_matching\base\brier_port.tif"

st = time.time()

scale = 50
# sift
sift = cv2.SIFT_create()
# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

test_img = cv2.imdecode(np.fromfile(test_img_path, dtype=np.uint8), 0)

test_img_r = cv2.resize(test_img, (test_img.shape[0] // scale, test_img.shape[1] // scale))
# test_img_r = cv2.cvtColor(test_img_r, cv2.COLOR_BGR2GRAY)

inter = time.time()

with  rasterio.open(base_img_path) as ds_b:
    base_img = resample_raster(ds_b, scale=1 / scale)
    base_img_r = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

big_inter = time.time()

keypoints_t, descriptors_t = sift.detectAndCompute(test_img_r, None)
keypoints_b, descriptors_b = sift.detectAndCompute(base_img_r, None)

# feature matching
matches = bf.match(descriptors_b, descriptors_t)
matches = sorted(matches, key=lambda x: x.distance)

# queryIdx代表的特征点序列是keypoints_b中的，trainIdx代表的特征点序列是keypoints_t中的，此时这两张图中的特征点相互匹配
img3 = cv2.drawMatches(base_img_r, keypoints_b, test_img_r, keypoints_t, matches[:20], test_img_r,
                       flags=2)
print(inter - st)
print(big_inter - inter)
end = time.time()
print("time cost is  {}".format(end - st))

plt.imshow(img3)
plt.show()
