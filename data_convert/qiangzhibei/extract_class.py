import os, shutil


def classify_ship():
    data_dir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\舰船编号"
    outdir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\outship_cls"

    os.makedirs(outdir, exist_ok=True)

    suffix_list = ['jpg', 'png']
    for root, dir, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.split('.')[-1] not in suffix_list:
                print("file format isn't target  %s" % file_path)
                continue

            if "=" in file:
                cls_name = file.split('.')[0].split('=')[-1]
                file_out_dir = os.path.join(outdir, cls_name)
                os.makedirs(file_out_dir, exist_ok=True)
                new_name = "%s-%s" % (os.path.basename(os.path.dirname(file_path)), file)
                dst_path = os.path.join(file_out_dir, new_name)
                if os.path.exists(dst_path):
                    raise ("the file has existed  in path  %s" % dst_path)
                shutil.copy(file_path, dst_path)
            else:
                print('there is no = in  %s' % file_path)
                continue


def file_num_valid():
    data_dir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\舰船编号"
    outdir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\outship_cls"

    data_num = 0
    out_num = 0

    def loop_dir(num, target_dir):
        suffix_list = ['jpg', 'png']
        for root, dir, files in os.walk(target_dir):
            for file in files:
                if file.split('.')[-1] in suffix_list:
                    num += 1
        return num

    data_num = loop_dir(data_num, data_dir)
    out_num = loop_dir(out_num, outdir)

    print("data num is  %d, out num is %d" % (data_num, out_num))


def static_cls_num():
    "统计各类别下面图片数量"
    data_dir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\outship_cls"
    cls_names = os.listdir(data_dir)
    static_dict = {}

    for dir in cls_names:
        static_dict[dir] = len(os.listdir(os.path.join(data_dir, dir)))

    print(static_dict)


def image_num_name():
    "文件夹下图片以从0开始的数字为序命名"
    data_dir = r"D:\BaiduNetdiskDownload\强智杯-20210814-军舰军机-训练数据\outship_cls_num"

    dir_list = os.listdir(data_dir)

    for dir in dir_list:
        dir_path = os.path.join(data_dir, dir)
        files = os.listdir(dir_path)

        for num in range(len(files)):
            file_name = files[num]

            new_name = "%d.jpg" % num
            new_path = os.path.join(dir_path, new_name)
            if os.path.exists(new_path):
                raise ("the file has existed  %s" % new_path)
            shutil.move(os.path.join(dir_path, file_name), new_path)


# classify_ship()
# file_num_valid()
# static_cls_num()
image_num_name()
