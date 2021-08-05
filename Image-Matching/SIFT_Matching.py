import cv2, os, time, math, rasterio, sys
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window


def view_bar(num, total):
    """
    进度条
    :param num: 当前进度
    :param total: 任务总量
    :return:
    """
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%,%d' % (">" * rate_num, "-" * (100 - rate_num), rate_num, num)
    sys.stdout.write(r)
    sys.stdout.flush()


test_img_dir = r"E:\image_matching\test"
base_img_dir = r"E:\image_matching\base"
save_dir = r"E:\image_matching"

test_imgs = os.listdir(test_img_dir)
base_imgs = os.listdir(base_img_dir)
test_imgs = list(filter(lambda x: x.endswith('tif'), test_imgs))
base_imgs = list(filter(lambda x: x.endswith('tif'), base_imgs))

time_costs = []
st = time.time()

# sift
sift = cv2.SIFT_create()
# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

test_start = (7500, 4000)  # 从test image 中裁图的左上角行列
test_tile_size = 1000  # test image 裁出tile 的尺寸
base_tile_size = 4000  # 在base image 上滑动窗口的尺寸

for _t in test_imgs:
    test_img_path = os.path.join(test_img_dir, _t)

    with rasterio.open(test_img_path) as ds_t:
        # 起始列行，取的宽高
        _block = ds_t.read(window=Window(test_start[1], test_start[0], test_tile_size, test_tile_size))
        test_block = np.zeros([3, _block.shape[1], _block.shape[2]], dtype=np.uint8)
        test_block = _block[:3, :, :]
        test_block = np.ma.transpose(test_block, [1, 2, 0])
        test_block = cv2.cvtColor(test_block, cv2.COLOR_BGR2GRAY)

        keypoints_t, descriptors_t = sift.detectAndCompute(test_block, None)

    for _b in base_imgs:
        match_right = False
        base_img_path = os.path.join(base_img_dir, _b)

        with rasterio.open(base_img_path) as ds_b:
            height_num = math.ceil(ds_b.height / base_tile_size)
            wid_num = math.ceil(ds_b.width / base_tile_size)

            process_num = 0
            for ph in range(height_num):

                start_h = ph * base_tile_size
                if (start_h + base_tile_size) > ds_b.height:
                    start_h = ds_b.height - base_tile_size

                for pw in range(wid_num):
                    start_w = pw * base_tile_size
                    if (start_w + base_tile_size) > ds_b.width:
                        start_w = ds_b.width - base_tile_size

                    _block_b = ds_b.read(window=Window(start_w, start_h, base_tile_size, base_tile_size))
                    base_block = np.zeros([3, _block_b.shape[1], _block_b.shape[2]], dtype=np.uint8)
                    base_block = _block_b[:3, :, :]
                    base_block = np.ma.transpose(base_block, [1, 2, 0])
                    base_block = cv2.cvtColor(base_block, cv2.COLOR_BGR2GRAY)

                    keypoints_b, descriptors_b = sift.detectAndCompute(base_block, None)

                    # feature matching
                    matches = bf.match(descriptors_b, descriptors_t)
                    matches = sorted(matches, key=lambda x: x.distance)

                    process_num += 1
                    view_bar(process_num, height_num * wid_num)

                    saved_matches = matches[:10]
                    all_dist = 0
                    for m in saved_matches:
                        all_dist += m.distance

                    if all_dist > 100:
                        continue

                    # queryIdx代表的特征点序列是keypoints_b中的，trainIdx代表的特征点序列是keypoints_t中的，此时这两张图中的特征点相互匹配
                    img3 = cv2.drawMatches(base_block, keypoints_b, test_block, keypoints_t, matches[:20], test_block,
                                           flags=2)
                    plt.imshow(img3)
                    # plt.show()
                    save_path = os.path.join(save_dir, "%s_%s_%d_%d_%d.tif" %
                                             (os.path.basename(test_img_path).split(".")[0],
                                              os.path.basename(base_img_path).split(".")[0],
                                              start_h, start_w, base_tile_size
                                              ), )
                    plt.savefig(save_path)

                    match_right = True
                    break

                if match_right:
                    break

end = time.time()

print("cost time is {}".format(end - st))
