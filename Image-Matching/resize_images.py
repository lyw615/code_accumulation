import cv2, os, time
import numpy as np

"""
基准影像平均8秒一张
测试影像平均3秒一张
"""
st = time.time()
image_dir = [
    # r"D:\BaiduNetdiskDownload\基准影像&测试影像示例\测试影像",
    r"D:\BaiduNetdiskDownload\基准影像&测试影像示例\基准影像",
]

out_dir = r"E:\image_matching\resize_dir_b"
os.makedirs(out_dir, exist_ok=True)

resize_ratio = 50  # resize到原来的多少分之一

for dir in image_dir:
    images = os.listdir(dir)
    images = list(filter(lambda x: x.endswith('tif'), images))

    for _ in range(len(images)):
        img_name = images[_]
        suffix = img_name.split('.')[-1]
        img_path = os.path.join(dir, img_name)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        img = cv2.resize(img, (img.shape[0] // resize_ratio, img.shape[1] // resize_ratio))

        _out_dir = os.path.join(out_dir, os.path.basename(dir))
        os.makedirs(_out_dir, exist_ok=True)

        img_out_path = os.path.join(_out_dir, img_name)
        cv2.imencode('.%s' % suffix, img)[1].tofile(img_out_path)

end = time.time()
print("time cost is  %s" % (end - st))
