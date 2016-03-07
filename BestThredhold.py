# -*- coding: gb18030 -*-
__author__ = 'yu'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


plot_style = ['b', 'r', 'ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点


def plot_best_thredhold(file_name, thres_hold_num):
    file = open(file_name, 'r')

    x = np.zeros(thres_hold_num)
    y = np.zeros(thres_hold_num)
    line_cnt = 0
    for line in file:
        if line_cnt >= thres_hold_num:
            break
        x[line_cnt] = float(line_cnt)/1000
        y[line_cnt] = float(line.strip())

        line_cnt += 1
    file.close()
    plt.plot(x, y, plot_style[0])
    plt.show()
# plot_best_thredhold("E:\\实验\\异常点检测\\gsm_7d_6th阈值.txt")
# plot_best_thredhold("D:\\ComplexNetwork\\result\\best_threshold\\gsm_7d_6th阈值.txt", 600)
plot_best_thredhold("D:\\ComplexNetwork\\result\\best_threshold\\gsm_7d_6th阈值_分别sim.txt", 300)
# plot_best_thredhold("D:\\ComplexNetwork\\result\\best_threshold\\gsm_7_30.txt", 300)

