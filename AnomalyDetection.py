# -*- coding: gb18030 -*-
__author__ = 'yu'

import numpy as np
import scipy as sp
import time
import networkx as nx
import matplotlib.pyplot as plt
import MySQLdb


feature_num = 5
plot_style = ['b', 'r', 'ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点


def normalization_matrix(m):
    for col_index in range(0, m.shape[1]):
        max_val = max(m[:, col_index])
        min_val = min(m[:, col_index])
        if max_val == min_val:
            m[:, col_index] = 1
        else:
            m[:, col_index] = (m[:, col_index]-min_val) / (max_val-min_val)
    return m


#余弦相似度
def cos_similar(feature_1, feature_2):
    feature_1 = np.mat(feature_1)
    feature_2 = np.mat(feature_2)
    num = float(feature_1*feature_2.T)
    denom = np.linalg.norm(feature_1) * np.linalg.norm(feature_2)
    if denom == 0:
        return 0
    return 1-(num/denom)


def get_aba_gsm_node_feature(time_scale, num, days):
    table_name = 'aba_gsm_' + str(time_scale) + 'd'
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    cur = conn.cursor()

    sql_select = 'SELECT * FROM ' + table_name + ' WHERE num = %s'
    cur.execute(sql_select, num)
    result_data = cur.fetchall()
    # 特征矩阵
    node_feature = np.zeros((days, feature_num))
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
    conn.close()
    return node_feature


def aba_gsm_node_anomaly_detection(time_scale, num, days, window_size):
    similar_score = np.zeros((days))
    ab_line = np.zeros((days+5)) + 0.5
    node_feature = get_aba_gsm_node_feature(time_scale, num, days)
    normalization_feature = normalization_matrix(node_feature)

    begin_date = window_size
    end_date = normalization_feature.shape[0]
    for cur_date in range(begin_date, end_date):
        cur_feature = normalization_feature[cur_date]
        pre_feature = np.zeros((feature_num))
        for i in range(1, window_size + 1):
            pre_feature += normalization_feature[cur_date - i]
        # print pre_feature
        # print cur_feature
        similar = cos_similar(pre_feature, cur_feature)
        similar_score[cur_date] = similar
    print '111'
    print similar_score
    mean_val = np.mean(similar_score)
    var_val = np.var(similar_score)
    print mean_val, var_val
    cnt = 0
    for i in range(0, days):
        if similar_score[i] > 0.5:
            cnt += 1
            print i
            print similar_score[i]
    print cnt

    plt.plot(range(days), similar_score, plot_style[0])
    plt.plot(range(days+5), ab_line, plot_style[1])
    plt.show()

# aba_gsm_node_anomaly_detection(7, '13158418421', 82, 5)
# aba_gsm_node_anomaly_detection(7, '13154407822', 82, 5) #周期性
aba_gsm_node_anomaly_detection(7, '13086538278', 82, 5) #周期性


