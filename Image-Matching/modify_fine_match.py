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

time_costs = []
st = time.time()

# sift
sift = cv2.SIFT_create()
# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# 此时已经能够准确的确定图片之间的对应关系,从plot文件夹获取
relation_dir = r"E:\image_matching\out_dir"
plot_dir = os.path.join(relation_dir, 'match_plot')
txt_dir = os.path.join(relation_dir, 'txt')

rela_imgs = os.listdir(plot_dir)
rela_imgs = list(filter(lambda x: x.endswith('png'), rela_imgs))

for _rela in rela_imgs:
    name_b, name_t = _rela.split('.')[0].split('2')
    base_img_path = os.path.join(base_img_dir, name_b + '.tif')
    test_img_path = os.path.join(test_img_dir, name_t + '.tif')

    # 读取txt文件中的匹配区域
    txt_file = os.path.join(txt_dir, _rela.split('.')[0] + '.txt')
    with open(txt_file, 'r') as txt:
        rec_t = txt.readline().strip('\n')
        rec_t = rec_t.split(' ')[:4]

        rec_b = txt.readline().strip('\n')
        rec_b = rec_b.split(' ')[:4]
        rec_t = list(map(lambda x: int(x), rec_t))
        rec_b = list(map(lambda x: int(x), rec_b))

    with rasterio.open(base_img_path) as ds_b:
        _block_b = ds_b.read(window=Window(rec_b[0], rec_b[1], rec_b[2] - rec_b[0], rec_b[3] - rec_b[1]))

        base_block = _block_b[:3, :, :]
        base_block = np.ma.transpose(base_block, [1, 2, 0])
        base_block = cv2.cvtColor(base_block, cv2.COLOR_BGR2GRAY)

        keypoints_b, descriptors_b = sift.detectAndCompute(base_block, None)

        base_transform = ds_b.transform
        base_height = ds_b.height
        base_width = ds_b.width

        with rasterio.open(test_img_path) as ds_t:
            # 起始列行，取的宽高
            _block = ds_t.read(window=Window(rec_t[0], rec_t[1], rec_t[2] - rec_t[0], rec_t[3] - rec_t[1]))
            test_block = _block[:3, :, :]
            test_block = np.ma.transpose(test_block, [1, 2, 0])

            _height = ds_t.height
            _wid = ds_t.width

        test_block = cv2.cvtColor(test_block, cv2.COLOR_BGR2GRAY)

        keypoints_t, descriptors_t = sift.detectAndCompute(test_block, None)

        # feature matching
        matches = bf.match(descriptors_b, descriptors_t)
        matches = sorted(matches, key=lambda x: x.distance)

        # queryIdx代表的特征点序列是keypoints_b中的，trainIdx代表的特征点序列是keypoints_t中的，此时这两张图中的特征点相互匹配
        img3 = cv2.drawMatches(base_block, keypoints_b, test_block, keypoints_t, matches[:20], test_block,
                               flags=2)
        ee = time.time()
        print(ee - st)
        plt.imshow(img3)
        plt.show()
        # save_path = os.path.join(save_dir, "%s_%s.png" %
        #                          (os.path.basename(test_img_path).split(".")[0],
        #                           os.path.basename(base_img_path).split(".")[0],
        #                           ), )
        # plt.savefig(save_path)

        nearest_point = matches[0]
        qdx = nearest_point.queryIdx
        tdx = nearest_point.trainIdx
        point_b = list(keypoints_b[qdx].pt)  # (x,y),  up left as raw point
        point_t = list(keypoints_t[tdx].pt)

        # 这对距离最近的匹配点各自在原图上的位置
        point_b[0] += rec_b[0]  # 匹配点在基础影像上的像素坐标
        point_b[1] += rec_b[1]

        point_t[0] += rec_t[0]  # 匹配点在测试影像上的像素坐标
        point_t[1] += rec_t[1]

        # 得到测试影像的左上角原点在基础影像上的像素坐标
        test_raw_point_x = point_b[0] - point_t[0]
        test_raw_point_y = point_b[1] - point_t[1]

        # 测试影像右下角在基础影像上的像素坐标
        _right = test_raw_point_x + _wid
        _bot = test_raw_point_y + _height

        # 构建像素坐标转经纬度坐标的转换矩阵
        transform_metric = np.array(base_transform)
        transform_metric = np.reshape(transform_metric, (3, 3))

        # 测试影像左上角和右下角的经纬度坐标转换结果
        point_arr1 = np.array([_right, _bot, 1])
        point_arr2 = np.array([test_raw_point_x, test_raw_point_y, 1])
        geo_right, geo_bot, _ = np.matmul(transform_metric, point_arr1)
        geo_left, geo_top, _ = np.matmul(transform_metric, point_arr2)

        print("geo_right %f,geo_left %f,geo_bot %f,geo_top %f" % (geo_right, geo_left, geo_bot, geo_top))

        # TODO 是先把转换矩阵存入txt还是等bbox算出来之后再运行脚本得出目标的地理坐标，可以先存好，但是得把小数点后的位数精确好

        # #以此关键点作为测试影像在基础影像中的像素坐标的参考点
        # bbox=[200,110,500,699] #假装传入了一个测试影像上飞机的框 xmin,ymin,xmax,ymax
        #
        # #得到飞机中心点在基础影像中的像素坐标
        # bbox[0]+=test_raw_point_x   #得到飞机框在基础影像上的像素坐标
        # bbox[2]+=test_raw_point_x
        #
        # bbox[1] += test_raw_point_y
        # bbox[3] += test_raw_point_y
        #
        # #得到飞机中心点在基础影像中的影像坐标
        # fwd = Affine.from_gdal(*base_transform)
        # geo_xmin,geo_ymin=fwd * (bbox[0], bbox[1])
        # geo_xmax,geo_ymax=fwd * (bbox[2], bbox[3])

end = time.time()

print("cost time is {}".format(end - st))
