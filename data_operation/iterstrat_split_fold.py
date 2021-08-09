import argparse
import ast
import os, sys
import os.path as osp
import shutil

from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append("/home/data1/yw/competetion_tools/iterative-stratification")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def analyze_xml(file_name, max_col):
    '''
    从xml文件中解析class，对象位置
    :param file_name: xml文件位置
    :return: class，每个类别的矩形位置
    '''

    fp = open(file_name)

    class_name = []
    for p in fp:
        if '<object>' in p:
            bndbox = next(fp).split('>')[1].split('<')[0]  # 标准VOC会有这个字段
            name = next(fp).split('>')[1].split('<')[0]
            if name in class_name_dict.keys():
                class_name.append(class_name_dict[name])
            # class_name.append(next(fp).split('>')[1].split('<')[0])

    fp.close()

    "没有标签"
    if len(class_name) > 0:
        max_col = max(max_col, len(class_name))
        return [os.path.basename(file_name).split(".")[0]], class_name, max_col
    else:
        return [], [], []


def analyze_classes_proportition(xml_dir, xmls=None):
    '''
    从xml文件中解析class，对象位置
    :param file_name: xml文件位置
    :return: class，每个类别的矩形位置
    '''

    if not xmls:
        xmls = os.listdir(xml_dir)
    class_num_dict = {"holothurian": 0, "echinus": 0, "scallop": 0, "starfish": 0, "waterweeds": 0, }

    for xml in tqdm(xmls):
        xml_path = os.path.join(xml_dir, xml)

        fp = open(xml_path)

        for p in fp:
            if '<object>' in p:

                name = next(fp).split('>')[1].split('<')[0]
                if name in class_num_dict.keys():
                    class_num_dict[name] += 1

        fp.close()

    print(class_num_dict)

    # "没有标签"
    # if len(class_name) > 0:
    #     return  [os.path.basename(file_name).split(".")[0]],class_name
    # else:
    #     return [], []


def analyze_classes_wh_ratio(xml_dir, xmls=None, all_bbox=False):
    '''
    从xml文件中解析class，对象位置
    :param file_name: xml文件位置
    :return: class，每个类别的矩形位置
    '''
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ÉèÖÃ×ÖÌåÎªºÚÌå
    plt.rcParams['font.family'] = 'sans-serif'  # ÉèÖÃ×ÖÌåÑùÊ½
    plt.rcParams['figure.figsize'] = (10.0, 10.0)  # ÉèÖÃ×ÖÌå´óÐ¡

    if not xmls:
        xmls = os.listdir(xml_dir)
    class_ratio_dict = {"holothurian": [], "echinus": [], "scallop": [], "starfish": []}

    for xml in tqdm(xmls):
        xml_path = os.path.join(xml_dir, xml)

        fp = open(xml_path)

        name = None
        for p in fp:
            if '<object>' in p:
                name = next(fp).split('>')[1].split('<')[0]

            if '<bndbox>' in p:
                rectangle = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(4)]

                rectangle_w = rectangle[2] - rectangle[0]
                rectangle_h = rectangle[3] - rectangle[1]
                if name in class_ratio_dict.keys():
                    class_ratio_dict[name].append(round(rectangle_w / rectangle_h, 1))
        fp.close()

    if not all_bbox:
        for key in class_ratio_dict.keys():
            unique_ratio = set(class_ratio_dict[key])
            unique_ratio = sorted(unique_ratio)
            bbox_unique_count = [class_ratio_dict[key].count(i) for i in unique_ratio]

            wh_dataframe = pd.DataFrame(bbox_unique_count, index=unique_ratio, columns=["%s_wh_ratio" % key])
            wh_dataframe.plot(kind='bar', color="#55aacc")
            plt.show()
    else:
        all_bbox = []
        for key in class_ratio_dict.keys():
            all_bbox.extend(class_ratio_dict[key])
        unique_ratio = set(all_bbox)
        unique_ratio = sorted(unique_ratio)
        bbox_unique_count = [all_bbox.count(i) for i in unique_ratio]

        wh_dataframe = pd.DataFrame(bbox_unique_count, index=unique_ratio, columns=["%s_wh_ratio" % "all_classes"])
        wh_dataframe.plot(kind='bar', color="#55aacc")
        plt.show()


def get_imagesname_labels(xml_dir, xmls):
    if not xmls:
        xmls = os.listdir(xml_dir)

    X = []
    _Y = []

    max_col = 0
    for ind in tqdm(range(len(xmls))):
        xml_path = os.path.join(xml_dir, xmls[ind])
        x, y, _max_col = analyze_xml(xml_path, max_col)

        if len(x) > 0 and len(y) > 0:
            X.append(x)
            _Y.append(y)

            max_col = max(max_col, _max_col)

    Y = np.zeros(shape=(len(X), max_col), dtype=np.uint8)
    for _ in range(len(_Y)):
        Y[_][:len(_Y[_])] = _Y[_]

    return np.array(X), Y


def split_folds(folds_dir, annotation_dir, target_xml_names=None):
    X, Y = get_imagesname_labels(annotation_dir, xmls=target_xml_names)
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    os.makedirs(folds_dir, exist_ok=True)

    fold_num = 1
    for train_index, test_index in mskf.split(X, Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        # Y_train, Y_test = Y[train_index], Y[test_index]

        save_fold_to_csv(X_train, X_test, save_dir=os.path.join(folds_dir, "fold_v%s" % fold_num))
        fold_num += 1


def save_fold_to_csv(train_numpy_ob, test_numpy_ob, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    pd_data_train = pd.DataFrame(train_numpy_ob, columns=['filename'])
    save_path_train = os.path.join(save_dir, "train.csv")
    pd_data_train.to_csv(save_path_train)

    pd_data_test = pd.DataFrame(test_numpy_ob, columns=['filename'])
    save_path = os.path.join(save_dir, "test.csv")
    pd_data_test.to_csv(save_path)


def example_for_iterstrat():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0]])

    mskf = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    for train_index, test_index in mskf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


def check_class_balance():
    xml_dir = r"/home/data1/yw/water_detection/train/Annotations"

    for v_num in range(1, 6):
        xml_list = []
        test_xml = []

        csv_path = r"/home/data1/yw/water_detection/split_folds/fold_v%s/train.csv" % v_num
        test_csv = r"/home/data1/yw/water_detection/split_folds/fold_v%s/test.csv" % v_num
        with open(csv_path, "r") as f:
            for line_record in f.readlines():
                xml_name = line_record.strip("\n").split(',')[1] + ".xml"
                xml_list.append(xml_name)
            xml_list.pop(0)

        with open(test_csv, "r") as f:
            for line_record in f.readlines():
                xml_name = line_record.strip("\n").split(',')[1] + ".xml"
                test_xml.append(xml_name)
            test_xml.pop(0)

        print("fold_v%s" % v_num)
        analyze_classes_proportition(xml_dir, xmls=xml_list)
        analyze_classes_proportition(xml_dir, xmls=test_xml)
        print("\n")


def check_class_balance_txt():
    xml_dir = r"/home/data1/yw/water_detection/train/Annotations"

    for v_num in range(2, 3):
        xml_list = []
        test_xml = []

        csv_path = r"/home/data1/yw/water_detection/split_folds/fold_v2/fold_v%s/train.txt" % v_num
        test_csv = r"/home/data1/yw/water_detection/split_folds/fold_v2/fold_v%s/test.txt" % v_num
        with open(csv_path, "r") as f:
            for line_record in f.readlines():
                xml_name = line_record.strip("\n") + ".xml"
                xml_list.append(xml_name)

        with open(test_csv, "r") as f:
            for line_record in f.readlines():
                xml_name = line_record.strip("\n") + ".xml"
                test_xml.append(xml_name)

        print("fold_v%s" % v_num)
        analyze_classes_proportition(xml_dir, xmls=xml_list)
        analyze_classes_proportition(xml_dir, xmls=test_xml)
        print("\n")


def edit_csv(csv_dir):
    # csv_dir=r"/home/data1/yw/water_detection/split_folds/fold_v2"

    with open(os.path.join(csv_dir, "train.csv"), "r") as f:
        lines_record = f.readlines()
        lines_record.pop(0)

    with open(os.path.join(csv_dir, "train.txt"), "w") as f:
        for line in lines_record:
            f.write(line.strip("\n").split(',')[1] + "\n")

    with open(os.path.join(csv_dir, "test.csv"), "r") as f:
        lines_record = f.readlines()
        lines_record.pop(0)

    with open(os.path.join(csv_dir, "test.txt"), "w") as f:
        for line in lines_record:
            f.write(line.strip("\n").split(',')[1] + "\n")

    os.remove(os.path.join(csv_dir, "train.csv"))
    os.remove(os.path.join(csv_dir, "test.csv"))


def split_train_val(vfold_path, xml_dir, dirty_txt=None):
    train_val = os.path.join(vfold_path, "train.csv")  # 把要处理的xml文件名写入csv
    if os.path.exists(train_val):
        with open(train_val, 'r') as f:
            records = f.readlines()
            records.pop(0)

        target_xml_names = []
        for cord in records:
            target_xml_names.append(cord.strip("\n").split(',')[1] + '.xml')
    else:

        target_xml_names = os.listdir(xml_dir)
        target_xml_names = list(filter(lambda x: x.endswith('xml'), target_xml_names))

    if dirty_txt:  # there are dirty data in the given data, then filter them from txt
        with open(dirty_txt, 'r') as dt:
            dirty_list = dt.readlines()
        dirty_list = [x.strip('\n').split('.')[0] + '.xml' for x in dirty_list]

        target_xml_names = [x for x in target_xml_names if x not in dirty_list]

    split_folds(vfold_path, xml_dir, target_xml_names)

    # edit_csv(os.path.join(vfold_path,"fold_v2"))
    # shutil.copy(train_val,os.path.join(vfold_path,"fold_v2","trainval.txt"))


if __name__ == '__main__':
    # analyze_classes_proportition("/home/data1/yw/water_detection/train/Annotations")

    # split all dataset
    # split_folds(r"/home/data1/yw/mmdetection/data/water_detection/train_implmt/split_folds",r"/home/data1/yw/mmdetection/data/water_detection/train_implmt/Annotations")

    # class_name_dict={"holothurian":0 ,"echinus": 1,"scallop": 2,"starfish": 3,"waterweeds": 4,}
    # class_name_dict = {"echinus": 1, "scallop": 2, "starfish": 3, "holothurian": 4, }
    class_name_dict = {"Airport": 1, "Port": 2}

    # split train val dataset
    xml_dir = r"/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/train/Annotations"
    out_dir = r"/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold-v2"
    dirty_txt = r"/home/data1/yw/github_projects/personal_github/code_aculat/data_operation/image_pre_none.txt"
    split_train_val(out_dir, xml_dir, dirty_txt)
    # check_class_balance()
    # check_class_balance_txt()
    # edit_csv()
    # example_for_iterstrat()

    # analyse_image_wh("/home/data1/yw/water_detection/train/Annotations",
    #                  csv_path=["/home/data1/yw/water_detection/split_folds/fold_v1/train.csv",
    #                            "/home/data1/yw/water_detection/split_folds/fold_v1/test.csv"])
