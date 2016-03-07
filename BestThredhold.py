# -*- coding: gb18030 -*-
__author__ = 'yu'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


plot_style = ['b', 'r', 'ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # ��


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
# plot_best_thredhold("E:\\ʵ��\\�쳣����\\gsm_7d_6th��ֵ.txt")
# plot_best_thredhold("D:\\ComplexNetwork\\result\\best_threshold\\gsm_7d_6th��ֵ.txt", 600)
plot_best_thredhold("D:\\ComplexNetwork\\result\\best_threshold\\gsm_7d_6th��ֵ_�ֱ�sim.txt", 300)
# plot_best_thredhold("D:\\ComplexNetwork\\result\\best_threshold\\gsm_7_30.txt", 300)

