# -*- coding: gb18030 -*-
__author__ = 'yu'

import numpy as np
import scipy as sp
import time
import networkx as nx
import matplotlib.pyplot as plt
import MySQLdb
import os


# 1d last 530
# 7d last 75
# 30d last 17
aba_gsm_days = 530
feature_num = 5
plot_style = ['b', 'r', 'ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点
conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
current_path = "D:\\ComplexNetwork"


def normalization_matrix(m):
    for col_index in range(0, m.shape[1]):
        max_val = max(m[:, col_index])
        min_val = min(m[:, col_index])
        if max_val == min_val:
            m[:, col_index] = 1
        else:
            m[:, col_index] = (m[:, col_index]-min_val) / (max_val-min_val)
    return m


# 余弦相似度
def cos_similar(feature_1, feature_2):
    feature_1 = np.mat(feature_1)
    feature_2 = np.mat(feature_2)
    num = float(feature_1*feature_2.T)
    denom = np.linalg.norm(feature_1) * np.linalg.norm(feature_2)
    if denom == 0:
        return 0
    return 1-(num/denom)


def aba_gsm_node_anomaly_detection_best_threhold(time_scale, num, slice_size, window_size, decay):
    similar_score = np.zeros((slice_size))
    ab_line = np.zeros((slice_size+5)) + 0.5
    node_feature = get_aba_gsm_node_feature(time_scale, num, slice_size)
    normalization_feature = normalization_matrix(node_feature)

    begin_date = 0
    end_date = normalization_feature.shape[0]
    for cur_date in range(begin_date, end_date):
        cur_feature = normalization_feature[cur_date]

        # for i in range(1, min(cur_date, window_size + 1)):
        #     pre_feature += normalization_feature[cur_date - i] * pow(decay, i)
        # pre_feature /= window_size
        # similar = cos_similar(pre_feature, cur_feature)
        if cur_date == begin_date:
            continue
        similar = 0.0
        windows_cnt = 0
        for i in range(1, min(cur_date + 1, window_size + 1)):
            similar += cos_similar(normalization_feature[cur_date - i], cur_feature) * pow(decay, i)
            windows_cnt += 1
        similar /= windows_cnt
        similar_score[cur_date] = similar

    return similar_score[slice_size-1]


def get_best_threhold(time_scale, slice_size, window_size, decay):
    table_name = 'aba_gsm_' + str(time_scale) + 'd'

    best_thres_hold_file_name = current_path + "\\result\\best_threshold\\gsm_" + str(time_scale) + "_" + str(slice_size) + ".txt"
    best_thres_hold_file = open(best_thres_hold_file_name, 'w')
    node_anomaly_score_file_name = current_path + "\\result\\node_anomaly_score\\gsm_" + str(time_scale) + "_" + str(slice_size) + ".txt"
    node_anomaly_score_file = open(node_anomaly_score_file_name, 'w')

    cur = conn.cursor()


    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index=%s'
    cur.execute(sql_select, slice_size)
    result_data = cur.fetchall()

    cnt = 0
    threhold_count = np.zeros((1000))
    for row in result_data:
        res = aba_gsm_node_anomaly_detection_best_threhold(time_scale, row[1], slice_size, window_size, decay)
        node_anomaly_score_file.write(row[1] + '    ' + str(res) + '\n')
        res *= 1000
        threhold_count[:int(res)] += 1
        cnt += 1
        print cnt

    cur.close()
    conn.commit()


    for threhold in threhold_count:
        best_thres_hold_file.write(str(threhold) + '\n')
    best_thres_hold_file.close()
    node_anomaly_score_file.close()


def get_aba_gsm_node_feature(time_scale, num, slice_size):
    table_name = 'aba_gsm_' + str(time_scale) + 'd'

    cur = conn.cursor()

    sql_select = 'SELECT * FROM ' + table_name + ' WHERE num = %s and date_index<=%s'
    cur.execute(sql_select, (num, slice_size))
    result_data = cur.fetchall()
    # 特征矩阵
    node_feature = np.zeros((slice_size, feature_num))
    for row in result_data:
        r_num = row[5] - 1
        node_feature[r_num][0] = row[2]             # 出度
        node_feature[r_num][1] = row[3]             # 入度
        node_feature[r_num][2] = row[4]             # 聚集系数
        node_feature[r_num][3] = row[2] + row[3]    # 总和
        node_feature[r_num][4] = (row[2]+0.1)/(row[3]+0.1)  # 出入度比
        # if r_num != 0:
        #     node_feature[r_num][5] = (node_feature[r_num][3] - node_feature[r_num-1][3]) /            # 趋势

    cur.close()
    conn.commit()

    return len(result_data), node_feature


def aba_gsm_node_anomaly_detection(time_scale, num, slice_size, window_size, decay, threshold):
    similar_score = np.zeros((slice_size))
    ab_line = np.zeros((slice_size+5)) + 0.25
    feature_num, node_feature = get_aba_gsm_node_feature(time_scale, num, slice_size)
    normalization_feature = normalization_matrix(node_feature)

    begin_date = 0
    end_date = normalization_feature.shape[0]
    for cur_date in range(begin_date, end_date):
        cur_feature = normalization_feature[cur_date]

        # for i in range(0, min(cur_date + 1, window_size + 1)):
        #     pre_feature += normalization_feature[cur_date - i] * pow(decay, i)
        # pre_feature /= window_size
        # similar = cos_similar(pre_feature, cur_feature)

        if cur_date == begin_date:
            continue
        similar = 0.0
        windows_cnt = 0
        for i in range(1, min(cur_date + 1, window_size + 1)):
            similar += cos_similar(normalization_feature[cur_date - i], cur_feature) * pow(decay, i)
            windows_cnt += 1
        similar /= windows_cnt
        if similar < 0.000001:
            similar = 0.0
        similar_score[cur_date] = similar

    return feature_num, similar_score
    # plt.plot(range(slice_size), similar_score, plot_style[2])
    # plt.plot(range(slice_size+5), ab_line, plot_style[1])
    # plt.show()


def get_node_anomaly_score(time_scale, slice_size):
    begin_line = 2092804
    num_file_name = current_path + "\\result\\num.txt"
    num_file = open(num_file_name, 'r')
    node_score_file_name = current_path + "\\result\\node_score\\gsm_" + str(time_scale) + "-" + str(begin_line) + ".txt"
    node_score_file = open(node_score_file_name, 'w')

    line_cnt = 0

    for line in num_file:
        line_cnt += 1
        if line_cnt == begin_line:
            break

    for line in num_file:
        line_cnt += 1
        print line_cnt
        num = line.strip()
        feature_num, score = aba_gsm_node_anomaly_detection(time_scale, num, slice_size, 5, 0.8, 0.25)
        node_score_file.write(num + " " + str(feature_num) + " ")
        for s in score:
            node_score_file.write(str(s))
            node_score_file.write(" ")
        node_score_file.write("\n")

    node_score_file.close()
    num_file.close()
    conn.close()
# get_best_threhold(7, 6, 5, 0.8)
# get_best_threhold(7, 10, 5, 0.8)
# get_best_threhold(7, 20, 5, 0.8)

# 找到7_6下大于0.25的点
# get_anomaly_node(7, 6, 5, 0.8, 0.25)
# get_best_threhold(30, 6, 5, 0.8)
# time_scale, num, slice_size, window_size, decay, threshold

# node_file_name = current_path + "\\result\\anomaly_node\\gsm_7_6.txt"
# node_file = open(node_file_name, 'r')
# for line in node_file:
#     aba_gsm_node_anomaly_detection(7, line.strip(), 76, 5, 0.8, 0.25)
# "\\result\\num.txt"

get_node_anomaly_score(30, 18)
