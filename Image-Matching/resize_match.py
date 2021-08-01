import cv2, time
import numpy as np
from matplotlib import pyplot as plt

"""
resize 50分之一后，大概6秒一张
对两个文件夹内的基准影像和测试影像遍历进行粗匹配，使用多进程加速
"""


test_img_path = r"E:\image_matching\test\布莱尔军港lit.tif"
base_img_path = r"E:\image_matching\base\brier_port.tif"

st = time.time()

# sift
sift = cv2.SIFT_create()
# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

test_img = cv2.imdecode(np.fromfile(test_img_path, dtype=np.uint8), 1)
base_img = cv2.imdecode(np.fromfile(base_img_path, dtype=np.uint8), 1)

test_img_r = cv2.resize(test_img, (test_img.shape[0] // 50, test_img.shape[1] // 50))
test_img_r = cv2.cvtColor(test_img_r, cv2.COLOR_BGR2GRAY)

base_img_r = cv2.resize(base_img, (base_img.shape[0] // 50, base_img.shape[1] // 50))
base_img_r = cv2.cvtColor(base_img_r, cv2.COLOR_BGR2GRAY)

keypoints_t, descriptors_t = sift.detectAndCompute(test_img_r, None)
keypoints_b, descriptors_b = sift.detectAndCompute(base_img_r, None)

# feature matching
matches = bf.match(descriptors_b, descriptors_t)
matches = sorted(matches, key=lambda x: x.distance)

# queryIdx代表的特征点序列是keypoints_b中的，trainIdx代表的特征点序列是keypoints_t中的，此时这两张图中的特征点相互匹配
img3 = cv2.drawMatches(base_img_r, keypoints_b, test_img_r, keypoints_t, matches[:20], test_img_r,
                       flags=2)

end = time.time()
print("time cost is  {}".format(end - st))

plt.imshow(img3)
plt.show()
