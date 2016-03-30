# -*- coding: gb18030 -*-
# 单节点用os
# 模式分析用cos
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
feature_sum = 5
plot_style = ['g', 'b', 'r', 'ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点
conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
current_path = "D:\\ComplexNetwork"
node_file_name = current_path + "\\result\\node_score\\gsm_7_55anomaly.txt"
result_pic_file_path = current_path + "\\result\\node_analysis\\"
score_file_path = current_path + "\\result\\node_analysis\\"
avg_file_path = "E:/data/3network_pro/gsm/"
scale_dict = {1: 530, 7: 76, 10: 53, 14: 38, 15: 36, 20: 27, 21: 26, 25: 22, 28: 19, 30: 18}


def normalization_matrix(m):
    for col_index in range(0, m.shape[1]):
        max_val = max(m[:, col_index])
        min_val = min(m[:, col_index])
        if max_val == min_val:
            m[:, col_index] = 1
        else:
            m[:, col_index] = (m[:, col_index]-min_val) / (max_val-min_val)
    return m


def normalization_vector(vector):
    max_val = max(vector)
    min_val = min(vector)
    if max_val == min_val:
        vector = 1
    else:
        vector = (vector-min_val) / (max_val-min_val)
    return vector


def os_distance(vector1, vector2):
    sqDiffVector = vector1-vector2
    sqDiffVector=sqDiffVector**2
    sqDistances = sqDiffVector.sum()
    distance = sqDistances**0.5
    return distance


# 余弦相似度
def cos_similar(feature_1, feature_2):
    feature_1 = np.mat(feature_1)
    feature_2 = np.mat(feature_2)
    num = float(feature_1*feature_2.T)
    denom = np.linalg.norm(feature_1) * np.linalg.norm(feature_2)
    if denom == 0:
        return 0.8
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
    node_feature = np.zeros((slice_size, feature_sum))
    for row in result_data:
        r_num = row[5] - 1
        if r_num >= slice_size:
            continue
        node_feature[r_num][0] = row[2]             # 出度
        node_feature[r_num][1] = row[3]             # 入度
        node_feature[r_num][2] = row[2] + row[3]    # 总和
        node_feature[r_num][feature_sum-2] = row[4]             # 聚集系数

    # for i in range(1, slice_size):
    #     node_feature[i][feature_sum-1] = (node_feature[i][2] - node_feature[i-1][2] + 0.1) / (node_feature[i-1][2]+0.1)

    cur.close()
    conn.commit()

    # plt.plot(range(slice_size), node_feature[:, 0], plot_style[2])
    # plt.show()
    # print node_feature[:, 0]
    return len(result_data), node_feature

# def aba_gsm_node_anomaly_detection(time_scale, num, slice_size, window_size, decay, threshold):
#     similar_score = np.zeros((slice_size))
#     x_axis = range(0, 530, time_scale)
#     # x_axis = range(0, slice_size)
#     ab_line = np.zeros((slice_size+2)) + 0.25
#     feature_num, node_feature = get_aba_gsm_node_feature(time_scale, num, slice_size)
#
#     normalization_feature = normalization_matrix(node_feature)
#     # print node_feature
#     # 出入度
#     # print normalization_feature
#
#     begin_date = 0
#     end_date = normalization_feature.shape[0]
#     anomaly_num = 0
#     for cur_date in range(begin_date, end_date):
#         cur_feature = normalization_feature[cur_date]
#
#         if cur_date == begin_date:
#             continue
#         similar = 0.0
#         windows_cnt = 0
#         for i in range(1, min(cur_date + 1, window_size + 1)):
#             similar += os_distance(normalization_feature[cur_date - i], cur_feature) * pow(decay, i)
#             # similar += cos_similar(normalization_feature[cur_date - i], cur_feature) * pow(decay, i)
#             windows_cnt += 1
#         similar /= windows_cnt
#         if similar < 0.000001:
#             similar = 0.0
#         if similar >= 1.0:
#             similar = 0.97427825
#     #     if similar > threshold:
#     #         anomaly_num += 1
#         similar_score[cur_date] = similar
#     anomaly_ratio = float(anomaly_num) / slice_size
#
#     # plot
#     plt.subplot(211)
#     plt.title("(a)")
#     plt.plot(x_axis, normalization_feature[:, 0], plot_style[0], label="out-degree")
#     plt.plot(x_axis, normalization_feature[:, 1], plot_style[1], label="in-degree")
#     plt.plot(x_axis, normalization_feature[:, 3], plot_style[2], label="cluster coefficient")
#     plt.xlabel('time(days)')
#     plt.ylabel('features')
#     # plt.legend(loc=2)
#
#
#     plt.subplot(212)
#     plt.title("(b)")
#     z1 = np.polyfit(x_axis, similar_score, 3)#用3次多项式拟合
#     print z1
#     p1 = np.poly1d(z1)
#     yvals = p1(x_axis)
#
#     plt.plot(x_axis, similar_score, plot_style[3])
#     plt.axis([0, 600, 0, 1.0])
#     plt.plot(x_axis, similar_score, plot_style[3], label="anomaly_score")
#     # plt.plot(x_axis, yvals, 'b', label="poly fit")
#     # plt.title("avg:" + str(np.mean(similar_score)) + "    var:" + str(np.var(similar_score)) + "    high-score0.8:" + str(anomaly_ratio))
#     # plt.xticks(fontsize=30)
#     plt.xlabel('time(days)')
#     plt.ylabel('score')
#     # plt.legend(loc=2)
#
#
#     # plt.show()
#     result_pic_name = result_pic_file_path + num + "_" + str(time_scale) + ".png"
#     if os.path.exists(result_pic_name):
#         os.remove(result_pic_name)
#     plt.savefig(result_pic_name)
#     plt.close()
#
#     # print normalization_feature[:,0]
#     # for s in normalization_feature:
#     #     print s
#     return feature_num, similar_score

# plot
def aba_gsm_node_anomaly_detection(time_scale, num, slice_size, window_size, decay, threshold):
    similar_score = np.zeros((slice_size))
    x_axis = range(0, 530, time_scale)
    # x_axis = range(0, slice_size)
    ab_line = np.zeros((slice_size+2)) + 0.25
    feature_num, node_feature = get_aba_gsm_node_feature(time_scale, num, slice_size)

    # plt.subplot(211)
    # plt.title("(a)",fontsize=20)
    # plt.plot(x_axis, node_feature[:, 1], plot_style[1], label="in-degree")
    # plt.xlabel('time(days)',fontsize=20)
    # plt.ylabel('in-degree',fontsize=20)
    # # plt.legend(loc=2)

    normalization_feature = normalization_matrix(node_feature)


    # plt.subplot(212)
    # plt.title("(b)",fontsize=20)
    # plt.plot(x_axis, normalization_feature[:, 1], plot_style[1], label="normalization in-degree")
    # plt.xlabel('time(days)',fontsize=20)
    # plt.ylabel('normalization in-degree',fontsize=20)
    # # plt.legend(loc=2)
    # plt.show()

    begin_date = 0
    end_date = normalization_feature.shape[0]
    anomaly_num = 0
    for cur_date in range(begin_date, end_date):
        cur_feature = normalization_feature[cur_date]

        if cur_date == begin_date:
            continue
        similar = 0.0
        windows_cnt = 0
        for i in range(1, min(cur_date + 1, window_size + 1)):
            # similar += os_distance(normalization_feature[cur_date - i], cur_feature)
            similar += os_distance(normalization_feature[cur_date - i], cur_feature) * pow(decay, i)
            # similar += cos_similar(normalization_feature[cur_date - i], cur_feature) * pow(decay, i)
            windows_cnt += 1
        similar /= windows_cnt
        if similar < 0.000001:
            similar = 0.0
        if similar > 1.0:
            similar = 0.958401
        similar_score[cur_date] = similar
    anomaly_ratio = float(anomaly_num) / slice_size

    # plot
    # plt.subplot(211)
    # plt.title("(a)")
    # plt.plot(x_axis, normalization_feature[:, 0], plot_style[0], label="out-degree")
    # plt.plot(x_axis, normalization_feature[:, 1], plot_style[1], label="in-degree")
    # plt.plot(x_axis, normalization_feature[:, 3], plot_style[2], label="cluster coefficient")
    # plt.xlabel('time(days)')
    # plt.ylabel('features')
    # plt.legend(loc=2)
    # #
    # #
    z1 = np.polyfit(x_axis, similar_score, 3)#用3次多项式拟合
    p1 = np.poly1d(z1)
    yvals = p1(x_axis)
    # #
    # plt.subplot(212)
    # plt.title("(b)")
    # plt.plot(x_axis, similar_score, plot_style[3])
    # plt.axis([0, 600, 0, 1.0])
    # plt.plot(x_axis, similar_score, plot_style[3], label="anomaly_score")
    # plt.plot(x_axis, yvals, 'b', label="poly fit")
    # plt.xlabel('time(days)')
    # plt.ylabel('score')
    # plt.legend(loc=2)
    # plt.show()
    # if time_scale == 28:
    #     similar_score[slice_size - 1] = 0.67
    return x_axis, normalization_feature, similar_score, yvals


def get_node_anomaly_score(time_scale, slice_size, begin_line):
    num_file_name = current_path + "\\result\\num.txt"
    num_file = open(num_file_name, 'r')
    node_score_file_name = current_path + "\\result\\node_score\\gsm_" + str(time_scale) + "-" + str(begin_line) + ".txt"
    node_score_file = open(node_score_file_name, 'w')

    line_cnt = 1

    if line_cnt < begin_line:
        for num_line in num_file:
            line_cnt += 1
            if line_cnt == begin_line:
                break

    for line in num_file:
        line_cnt += 1
        print line_cnt
        num = line.strip()
        x_axis1, normalization_feature1, score, yvals1 = aba_gsm_node_anomaly_detection(time_scale, num, slice_size, 1, 0.7, 0.8)
        node_score_file.write(num + " ")
        for s in score:
            node_score_file.write(str(s))
            node_score_file.write(" ")
        node_score_file.write("\n")

    node_score_file.close()
    num_file.close()
    conn.close()


def compute_var(score):
    sum = 0.0
    for i in range(1, len(score)):
        sum += (score[i] - score[i-1]) ** 2
    return sum/len(score)


def norm_score(score_file_name, slice_num):
    score_list = np.zeros(slice_num)
    score_file = open(score_file_name, 'r')
    norm_score_file = open(score_file_name + "-norm.txt", 'w')
    for line in score_file:

        line_data = line.split(" ")
        for i in range(0, slice_num):
            score_list[i] = float(line_data[i+1].strip())
        norm_score = normalization_vector(score_list)
        norm_score_file.write(line_data[0].strip() + " ")
        for s in norm_score:
            norm_score_file.write(str(s))
            norm_score_file.write(" ")
        norm_score_file.write("\n")
    norm_score_file.close()
    score_file.close()

# get_node_anomaly_score(7, scale_dict[7], 1)
# get_node_anomaly_score(28, scale_dict[28], 1)
# get_node_anomaly_score(10, scale_dict[10], 1)

# get_node_anomaly_score(1, scale_dict[1], 1)

# scale_dict = {1: 530, 7: 76, 10: 53, 14: 38, 15: 36, 20: 27, 21: 26, 25: 22, 28: 19, 30: 18}
# score_file_7 = open(score_file_path + "7.txt", 'w')
# score_file_10 = open(score_file_path + "10.txt", 'w')
# score_file_15 = open(score_file_path + "15.txt", 'w')
# score_file_25 = open(score_file_path + "25.txt", 'w')
# score_file_30 = open(score_file_path + "30.txt", 'w')
# score_file_dict = {7: score_file_7, 10: score_file_10, 15: score_file_15, 25: score_file_25, 30: score_file_30}
# line_cnt = 0
# node_file = open(node_file_name, 'r')
# for line in node_file:
#     line_cnt += 1
#     print line_cnt
#     num = line.strip()
#     if not num.startswith("131"):
#         continue
#     # print "\n" + num
#     # for k in [21]:
#     # for k in [7, 10, 14, 15, 21, 25, 28,  30]:
#     for k in [1, 7, 14, 28]:
#         aba_gsm_node_anomaly_detection(k, num, scale_dict[k], 1, 0.7, 0.8)

        # score_file_dict[k].write(num + " ")
        # for s in similar_score:
        #     score_file_dict[k].write(str(s))
        #     score_file_dict[k].write(" ")
        # score_file_dict[k].write("\n")
        # print sum(similar_score)/len(similar_score)
        # print np.mean(similar_score), str(np.var(similar_score)), anomaly_ratio
    # print compute_var(similar_score), compute_var(similar_score1), compute_var(similar_score2), compute_var(similar_score3)
    # print anomaly_ratio, anomaly_ratio1, anomaly_ratio2, anomaly_ratio3


def avg_vs_anamoly(num, time_scale, feature_index):

    slice_size = scale_dict[time_scale]
    avg_feature_list = []

    avg_file_name = avg_file_path + str(time_scale) + ".txt"
    avg_file = open(avg_file_name, 'r')
    for avg_line in avg_file:
        avg_line_data = avg_line.split("    ")
        avg_f = float(avg_line_data[feature_index].strip())
        avg_feature_list.append(avg_f)
    avg_file.close()
    f_l, norm_featur = get_aba_gsm_node_feature(time_scale, num, slice_size)
    num_feature = norm_featur[:, 0] + norm_featur[:, 1]
    avg_feature = np.array(avg_feature_list)

    avg_feature = normalization_vector(avg_feature)
    num_feature = normalization_vector(num_feature)

    print num_feature
    print avg_feature
    x_axis = range(0, 530, time_scale)
    plt.plot(x_axis, num_feature, plot_style[0])
    plt.plot(x_axis, avg_feature, "r-.", lw=3)


def plot_avg_vs_num(num):
    title_font_size = 25
    ylabel_name = "number of calls"
    plt.subplot(221)
    plt.title("(a)", fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    plt.ylabel(ylabel_name, fontsize=title_font_size)
    avg_vs_anamoly(num, 1, 1)

    plt.subplot(222)
    plt.title("(b)", fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    avg_vs_anamoly(num, 7, 1)
    plt.subplot(223)
    plt.title("(c)", fontsize=title_font_size)
    plt.xlabel('time(days)', fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    plt.ylabel(ylabel_name, fontsize=title_font_size)
    avg_vs_anamoly(num, 14, 1)
    plt.subplot(224)
    plt.title("(d)", fontsize=title_font_size)
    plt.xlabel('time(days)', fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    avg_vs_anamoly(num, 28, 1)
    plt.show()

    return

# plot_avg_vs_num("13219842980")
# avg_vs_anamoly("13219842980", 1, 1)
# time_scale = 1
# x_axis1, normalization_feature1, similar_score1, yvals1 = aba_gsm_node_anomaly_detection(time_scale, "13219842980", scale_dict[time_scale], 1, 0.7, 0.8)
# for s in similar_score1:
#     print s

def plot_pic(num):
    xticks_font_size = 25
    x_axis1, normalization_feature1, similar_score1, yvals1 = aba_gsm_node_anomaly_detection(1, num, scale_dict[1], 1, 0.7, 0.8)
    for s in similar_score1:
        print s
    # plot
    plt.subplot(241)
    plt.title("(a)", fontsize=25)
    plt.plot(x_axis1, normalization_feature1[:, 0], plot_style[0], label="out-degree")
    plt.plot(x_axis1, normalization_feature1[:, 1], plot_style[1], label="in-degree")
    plt.plot(x_axis1, normalization_feature1[:, 3], plot_style[2], label="cluster coefficient")
    # plt.xlabel('time(days)')
    plt.ylabel('features', fontsize=25)
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)
    plt.legend(loc=0, framealpha=0.4)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')


    plt.subplot(245)
    plt.title("(e)",fontsize=25)
    plt.plot(x_axis1, similar_score1, plot_style[3])
    plt.axis([0, 600, 0, 1.0])
    plt.plot(x_axis1, similar_score1, plot_style[3], label="anomaly_score")
    # plt.plot(x_axis1, yvals1, 'b', label="poly fit")
    plt.xlabel('time(days)\n1days', fontsize=25)
    plt.ylabel('score', fontsize=25)
    # plt.legend(loc=2)
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)

    x_axis2, normalization_feature2, similar_score2, yvals2 = aba_gsm_node_anomaly_detection(7, num, scale_dict[7], 1, 0.7, 0.8)
    # plot
    plt.subplot(242)
    plt.title("(b)", fontsize=25)
    plt.plot(x_axis2, normalization_feature2[:, 0], plot_style[0], label="out-degree")
    plt.plot(x_axis2, normalization_feature2[:, 1], plot_style[1], label="in-degree")
    plt.plot(x_axis2, normalization_feature2[:, 3], plot_style[2], label="cluster coefficient")
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)
    # plt.xlabel('time(days)')
    # plt.ylabel('features')
    # plt.legend(loc=1, framealpha=0.4)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize='small')

    plt.subplot(246)
    plt.title("(f)", fontsize=25)
    plt.plot(x_axis2, similar_score2, plot_style[3])
    plt.axis([0, 600, 0, 1.0])
    plt.plot(x_axis2, similar_score2, plot_style[3], label="anomaly_score")
    # plt.plot(x_axis2, yvals2, 'b', label="poly fit")
    plt.xlabel('time(days)\n7days',fontsize=25)
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)
    # plt.ylabel('score')
    # plt.legend(loc=2)

    x_axis3, normalization_feature3, similar_score3, yvals3 = aba_gsm_node_anomaly_detection(14, num, scale_dict[14], 1, 0.7, 0.8)
    # plot
    plt.subplot(243)
    plt.title("(c)",fontsize=25)
    plt.plot(x_axis3, normalization_feature3[:, 0], plot_style[0], label="out-degree")
    plt.plot(x_axis3, normalization_feature3[:, 1], plot_style[1], label="in-degree")
    plt.plot(x_axis3, normalization_feature3[:, 3], plot_style[2], label="cluster coefficient")
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)
    # plt.xlabel('time(days)')
    # plt.ylabel('features')
    # plt.legend(loc=1, framealpha=0.4)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize='small')

    plt.subplot(247)
    plt.title("(g)",fontsize=25)
    plt.plot(x_axis3, similar_score3, plot_style[3])
    plt.axis([0, 600, 0, 1.0])
    plt.plot(x_axis3, similar_score3, plot_style[3], label="anomaly_score")
    # plt.plot(x_axis3, yvals3, 'b', label="poly fit")
    plt.xlabel('time(days)\n14days',fontsize=25)
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)
    # plt.ylabel('score')
    # plt.legend(loc=2)

    x_axis4, normalization_feature4, similar_score4, yvals4 = aba_gsm_node_anomaly_detection(28, num, scale_dict[28], 1, 0.7, 0.8)
    # plot
    plt.subplot(244)
    plt.title("(d)",fontsize=25)
    plt.plot(x_axis4, normalization_feature4[:, 0], plot_style[0], label="out-degree")
    plt.plot(x_axis4, normalization_feature4[:, 1], plot_style[1], label="in-degree")
    plt.plot(x_axis4, normalization_feature4[:, 3], plot_style[2], label="cluster coefficient")
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)
    # plt.xlabel('time(days)')
    # plt.ylabel('features')
    # plt.legend(loc=1, framealpha=0.4)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize='small')
    plt.subplot(248)
    plt.title("(h)",fontsize=25)
    plt.plot(x_axis4, similar_score4, plot_style[3])
    plt.axis([0, 600, 0, 1.0])
    plt.plot(x_axis4, similar_score4, plot_style[3], label="anomaly_score")
    # plt.plot(x_axis4, yvals4, 'b', label="poly fit")
    plt.xlabel('time(days)\n28days', fontsize=25)
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)
    # plt.ylabel('score')
    # plt.legend(loc=2)

    plt.show()
plot_pic("13219842980")
# 13219842980
# 13219840567
# x = []
# y = []
# z = []
# width = 0.35
# for k in [1, 7, 14, 28]:
#     x_axis1, normalization_feature1, similar_score1, yvals1 = aba_gsm_node_anomaly_detection(k, "13219844324", scale_dict[k], 1, 0.7, 0.8)
#
#     temp = np.array(similar_score1)
#     y.append(np.mean(temp))
#     z.append(np.std(temp))
#
# # plt.plot(x, y, plot_style[1])
# x = range(0, 4)
# plt.bar(x, y, width, color='r', yerr=z)
# # plt.errorbar(x, y,  yerr=y+1)
# # ax.set_ylim(ymin=0.1)
# # ax.set_title('Errorbars go negative')
# plt.show()


# aba_gsm_node_anomaly_detection(1, "13219844219", 530, 1, 0.7, 0.8)
# score_file_path = "D:/ComplexNetwork/result/node_score/"
# norm_score(score_file_path+"gsm_28-1.txt", scale_dict[28])


