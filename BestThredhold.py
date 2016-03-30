# -*- coding: gb18030 -*-
__author__ = 'yu'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


plot_style = ['b', 'r', 'ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点
scale_dict = {1: 530, 7: 76, 10: 53, 14: 38, 15: 36, 20: 27, 21: 26, 25: 22, 28: 19, 30: 18}

def normalization_vector(vector):
    max_val = max(vector)
    min_val = min(vector)
    if max_val == min_val:
        vector = 1
    else:
        vector = (vector-min_val) / (max_val-min_val)
    return vector
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
# plot_best_thredhold("D:\\ComplexNetwork\\result\\best_threshold\\gsm_7d_6th阈值_分别sim.txt", 300)
# plot_best_thredhold("D:\\ComplexNetwork\\result\\best_threshold\\gsm_7_30.txt", 300)

def get_best_thredhold(file_name, date_index):
    thres_num = 1000
    file = open(file_name, 'r')
    x = []
    y = []
    thres = np.zeros(thres_num)
    index1_cnt = 0
    line_cnt = 0
    for line in file:
        line_data = line.split(" ")
        score = float(line_data[date_index]) / 1.3
        # score = float(line_data[date_index]) / 2
        x.append(score)
    file.close()

    # norm_x_array = normalization_vector(np.array(x))
    norm_x_array = np.array(x)
    for norm_x in norm_x_array:
        index = int(norm_x * thres_num)
        thres[:index] += 1

    for thre in thres:
        print thre


    #
    fig, bx = plt.subplots(1, 1)
    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))

    thres_x = []
    for i in range(0, thres_num):
        thres_x.append((float(i)/thres_num))
    plt.plot(thres_x, thres, plot_style[1])
    bx.yaxis.set_major_formatter(formatter)
    plt.xlabel('$\\varepsilon$', fontsize=35)
    plt.ylabel('anomaly amount', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()


def get_every_time_anomaly_cnt(file_name, slice_size, threshold):
    anomaly_cnt = np.zeros(slice_size)
    file = open(file_name, 'r')

    for line in file:
        line_data = line.split(" ")
        for i in range(0, slice_size):
            score = float(line_data[i+1]) / 2
            if score > threshold:
                anomaly_cnt[i] += 1
    file.close()
    print anomaly_cnt


# get_every_time_anomaly_cnt("D:\\ComplexNetwork\\result\\node_score\\gsm_1-1.txt", scale_dict[1], 0.5)
get_best_thredhold("D:\\ComplexNetwork\\result\\node_score\\gsm_1-1.txt", 6)

# def anomaly_num(file_name, thres_hold, slice_size):
#     slice_num = np.zeros(slice_size)
#     file = open(file_name, 'r')
#     for line in file:
#         line_data = line.split(" ")
#         for i in range(2, slice_size+2):
#             if float(line_data[i]) > thres_hold:
#                 slice_num[i-2] += 1
#     file.close()
#     print slice_num
#
# anomaly_num("D:\\ComplexNetwork\\result\\node_score\\gsm_7-1.txt", 0.868, 76)
