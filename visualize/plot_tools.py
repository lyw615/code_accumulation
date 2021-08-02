# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np


def plot_points(point_list, label=None):
    """
    绘制散点图,  plot_points([(x1,y1),(x2,y2)])   or [[x1,y1],[x2,y2]]
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    plt.xlabel('X')
    plt.ylabel('Y')

    if len(point_list) > 0:
        for point in point_list:
            x1, y1 = point
            colors1 = '#00CED1'  # 点的颜色
            # colors2 = '#DC143C'  #如果再绘制一种点集，可以设置颜色同时绘制
            # area = np.pi * 4 ** 2  # 点面积

            # 画散点图
            plt.scatter(x1, y1, c=colors1, label=label)

            plt.legend()
            plt.show()
