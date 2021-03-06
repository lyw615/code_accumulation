import numpy as np


def analyze_xml(file_name, check_bbox=False):
    '''
    从xml文件中解析class，对象位置
    :param file_name: xml文件位置
    :return: class，每个类别的矩形位置
    '''
    fp = open(file_name)
    class_name = []
    rectangle_position = []
    for p in fp:
        if '<object>' in p:
            # bnbbox = next(fp)

            name = next(fp).split('>')[1].split('<')[0]
            class_name.append(name)

        elif '<bndbox>' in p:
            rectangle = []
            [rectangle.append(round(eval(next(fp).split('>')[1].split('<')[0]))) for _ in range(4)]
            rectangle_position.append(rectangle)
        elif '<size>' in p:
            size = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(2)]

    fp.close()
    if check_bbox:
        w, h = size
        if len(rectangle_position) > 0:
            _class_name = []
            rectangle_position = np.array(rectangle_position)
            # [xmin:xmax,ymin:ymax]没取到xmax,ymax
            rectangle_position[:, 2] = rectangle_position[:, 2]
            rectangle_position[:, 3] = rectangle_position[:, 3]

            rectangle_position[:, 2][rectangle_position[:, 2] > w - 1] = w - 1
            rectangle_position[:, 3][rectangle_position[:, 3] > h - 1] = h - 1

            rectangle_position[:, 0:3][rectangle_position[:, 0:3] < 0] = 0

            rectangle_position_w = rectangle_position[:, 2] - rectangle_position[:, 0]
            rectangle_position_h = rectangle_position[:, 3] - rectangle_position[:, 1]
            # 需要rectangle_position的宽高都大于1的rectangle_position才能进入下一步，and表示为与操作，两者都正确则结果正确
            right_index = np.logical_and(rectangle_position_w > 1, rectangle_position_h > 1)
            rectangle_position = (rectangle_position[right_index]).tolist()

            for id in range(len(right_index)):
                if right_index[id]:
                    _class_name.append(class_name[id])
            class_name = _class_name
    "没有标签"
    if len(rectangle_position) > 0:
        return class_name, rectangle_position
    else:
        return [], []

def analyze_obs_area_at_fix_scale(file_name, check_bbox=False):
    '''
    从xml文件中解析class，对象位置
    :param file_name: xml文件位置
    :return: class，每个类别的矩形位置
    '''
    fp = open(file_name)
    class_name = []
    rectangle_position = []
    for p in fp:
        if '<object>' in p:
            bnbbox = next(fp)

            name = next(fp).split('>')[1].split('<')[0]
            class_name.append(name)

        elif '<bndbox>' in p:
            rectangle = []
            [rectangle.append(round(eval(next(fp).split('>')[1].split('<')[0]))) for _ in range(4)]
            rectangle_position.append(rectangle)
        elif '<size>' in p:
            size = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(2)]

    fp.close()
    if check_bbox:
        w, h = size
        if len(rectangle_position) > 0:
            _class_name = []
            rectangle_position = np.array(rectangle_position)
            # [xmin:xmax,ymin:ymax]没取到xmax,ymax


            rectangle_position[:, 2][rectangle_position[:, 2] > w - 1] = w - 1
            rectangle_position[:, 3][rectangle_position[:, 3] > h - 1] = h - 1

            rectangle_position[:, 0:3][rectangle_position[:, 0:3] < 0] = 0

            rectangle_position_w = rectangle_position[:, 2] - rectangle_position[:, 0]
            rectangle_position_h = rectangle_position[:, 3] - rectangle_position[:, 1]
            # 需要rectangle_position的宽高都大于1的rectangle_position才能进入下一步，and表示为与操作，两者都正确则结果正确
            right_index = np.logical_and(rectangle_position_w > 1, rectangle_position_h > 1)
            rectangle_position = (rectangle_position[right_index]).tolist()

            for id in range(len(right_index)):
                if right_index[id]:
                    _class_name.append(class_name[id])
            class_name = _class_name
    "没有标签"
    if len(rectangle_position) > 0:
        return class_name, rectangle_position
    else:
        return [], []


def analyze_xml_hrsc2016(file_name, check_bbox=False):
    '''
    从xml文件中解析class，对象位置
    :param file_name: xml文件位置
    :return: class，每个类别的矩形位置
    '''
    id2name={"1": "航母","2": "两栖舰","15": "登陆舰","17": "支援舰", "21": "驱逐舰","22": "护卫舰","23": "巡洋舰"}
    # id2name={"1": "hangmu","2": "liangqi","15": "denglu","17": "zhiyuan", "21": "quzhu","22": "huwei","23": "xunyang"}
    fp = open(file_name)
    class_name = []
    rectangle_position = []
    for p in fp:
        if '<object>' in p:
            # bnbbox = next(fp)

            name = next(fp).split('>')[1].split('<')[0]
            class_name.append(id2name[name])

        elif '<bndbox>' in p:
            rectangle = []
            [rectangle.append(round(eval(next(fp).split('>')[1].split('<')[0]))) for _ in range(4)]
            rectangle_position.append(rectangle)
        elif '<size>' in p:
            size = [round(eval(next(fp).split('>')[1].split('<')[0])) for _ in range(2)]

    fp.close()
    if check_bbox:
        w, h = size
        if len(rectangle_position) > 0:
            _class_name = []
            rectangle_position = np.array(rectangle_position)
            # [xmin:xmax,ymin:ymax]没取到xmax,ymax
            rectangle_position[:, 2] = rectangle_position[:, 2] + 1
            rectangle_position[:, 3] = rectangle_position[:, 3] + 1

            rectangle_position[:, 2][rectangle_position[:, 2] > w - 1] = w - 1
            rectangle_position[:, 3][rectangle_position[:, 3] > h - 1] = h - 1

            rectangle_position[:, 0:3][rectangle_position[:, 0:3] < 0] = 0

            rectangle_position_w = rectangle_position[:, 2] - rectangle_position[:, 0]
            rectangle_position_h = rectangle_position[:, 3] - rectangle_position[:, 1]
            # 需要rectangle_position的宽高都大于1的rectangle_position才能进入下一步，and表示为与操作，两者都正确则结果正确
            right_index = np.logical_and(rectangle_position_w > 1, rectangle_position_h > 1)
            rectangle_position = (rectangle_position[right_index]).tolist()

            for id in range(len(right_index)):
                if right_index[id]:
                    _class_name.append(class_name[id])
            class_name = _class_name
    "没有标签"
    if len(rectangle_position) > 0:
        return class_name, rectangle_position
    else:
        return [], []
