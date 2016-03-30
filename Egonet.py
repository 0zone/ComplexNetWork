# -*- coding: gb18030 -*-
__author__ = 'yu'

import networkx as nx
import MySQLdb
import math
import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats as stats

current_path = "D:\\ComplexNetwork"
node_enw_file_name = current_path + "\\result\\ego_pro\\"
# node_enw_file_name = current_path + "\\result\\ego_pro\\sms\\"


def best_result():
    c = -0.01613327
    a = 1.11230972
    c1 = 0.01042072
    a1 = 1.06755459
    c2 = 0.01700159
    a2 = 1.05664055

    x = 116
    y = 118
    result_file_name = node_enw_file_name + "305-7.txt" + "_result--6.csv"
    result_file = open(result_file_name, 'w')
    en_file = open(node_enw_file_name + "305-7.txt", 'r')
    for line in en_file:
        data = line.split(' ')
        x = float(data[1])
        y = float(data[2])
        x1 = max(10**c1 * x**a1, y) / min(10**c1 * x**a1, y) * math.log(math.fabs(10**c1 * x**a1-y)+1)
        x2 = max(10**c2 * x**a2, y) / min(10**c2 * x**a2, y) * math.log(math.fabs(10**c2 * x**a2-y)+1)
        x3 = max(10**0.108 * x**1.3, y) / min(10**0.108 * x**1.3, y) * math.log(math.fabs(10**0.108 * x**1.3-y)+1)
        x4 = max(10**0.108 * x**1.4, y) / min(10**0.108 * x**1.4, y) * math.log(math.fabs(10**0.108 * x**1.4-y)+1)
        x5 = max(10**0.108 * x**1.5, y) / min(10**0.108 * x**1.5, y) * math.log(math.fabs(10**0.108 * x**1.5-y)+1)
        x6 = max((10**c) * (x**a), y) / min((10**c) * (x**a), y) * math.log(math.fabs((10**c) * (x**a)-y)+1)
        result_file.write(data[0] + "," + str(x6) + "," + str(x1) + "," + str(x2) + "," + str(x3) + "," + str(x4) + "," + str(x5) + "\n")
    en_file.close()
    result_file.close()


def trim_mean(arr, percent=0.1):
    # print arr
    arr.sort()
    # print arr
    n = len(arr)
    k = int(round((n*percent)/2))
    return np.mean(arr[k:n-k])


def drop_zeros(a_list):
    return [i for i in a_list if i > 0]


def log_binning(x, bin_count=10):
    min_x = math.log10(min(drop_zeros(x)))
    max_x = math.log10(max(drop_zeros(x)))
    bins = np.logspace(min_x, max_x, num=bin_count)

    return bins


def bining_data(x_vec, y_vec, bin_count=11):
    bins = log_binning(x_vec, bin_count)
    bin_x = [[] for i in range(bin_count-1)]
    bin_y = [[] for i in range(bin_count-1)]

    for x_index in range(0, len(x_vec)):
        x = x_vec[x_index]
        bin_index = 0
        for bin_index in range(0, bin_count):
            if bins[bin_index] > x:
                break
        bin_x[bin_index-1].append(x_vec[x_index])
        bin_y[bin_index-1].append(y_vec[x_index])
    return bin_x, bin_y


def get_gsm_network(begin_date, time_scale):
    table_name = 'aba_gsm'
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index >= %s and date_index < %s'

    select_cur = conn.cursor()
    select_cur.execute(sql_select, (begin_date, begin_date + time_scale))
    result_data = select_cur.fetchall()
    # 构建网络
    network = nx.Graph()
    for row in result_data:
        a = row[1]
        b = row[2]
        if not network.has_edge(a, b):
            network.add_edge(a, b, weight=1)
        else:
            network.add_edge(a, b, weight=1+network.get_edge_data(a, b)['weight'])
    select_cur.close()
    conn.commit()
    conn.close()
    return network


def get_sms_network(begin_date, time_scale):
    table_name = 'aba_sms'
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index >= %s and date_index < %s'

    select_cur = conn.cursor()
    select_cur.execute(sql_select, (begin_date, begin_date + time_scale))
    result_data = select_cur.fetchall()
    # 构建网络
    network = nx.Graph()
    for row in result_data:
        network.add_edge(row[1], row[2])
    select_cur.close()
    conn.commit()
    conn.close()
    return network


def get_network_from_file(network_file_name):
    network_file = open(network_file_name, 'r')

    network = nx.Graph()
    for line in network_file:
        line_data = line.split(" ")
        network.add_edge(line_data[0].strip(), line_data[1].strip(), weight=1)

    network_file.close()
    return network


def ego_stat(network, file_name):

    ego_pro_file = open(file_name, 'w')
    node_cnt = 0
    for node in network.nodes_iter():
        ego_net = nx.ego_graph(network, node, radius=1)
        n = ego_net.number_of_nodes() - 1
        if n <= 0:
            continue
        e = ego_net.number_of_edges() - ego_net.number_of_selfloops()
        w = sum(ego_net.degree(weight='weight').values()) / 2
        result = []
        result.append(node)
        result.append(str(n))
        result.append(str(e))
        result.append(str(w))
        result.append('\n')
        ego_pro_file.write(' '.join(result))
        node_cnt += 1
        print node_cnt
    ego_pro_file.close()


def iter_filter(n_vec, e_vec, a, c):
    # 去噪声
    thres = 10
    n_count = 0
    for x in n_vec:
        y = e_vec[n_count]
        if max((10**c) * (x**a), y) / min((10**c) * (x**a), y) * math.log(math.fabs((10**c) * (x**a)-y)+1) > thres:
            n_vec[n_count] = 1.0
            e_vec[n_count] = 1.0
        n_count += 1

    bin_count = 10
    bin_n_v, bin_e_v = bining_data(n_vec, e_vec, bin_count+1)
    med_n = np.zeros(bin_count)
    med_e = np.zeros(bin_count)
    for bin_index in range(0, bin_count):
        if len(bin_n_v[bin_index]) < 10:
            continue
        med_n[bin_index] = np.mean(bin_n_v[bin_index])
        med_e[bin_index] = np.mean(bin_e_v[bin_index])

    # 中位数拟合
    it_ls_fit = np.polyfit(np.log10(drop_zeros(med_n)), np.log10(drop_zeros(med_e)), 1)
    print it_ls_fit

    return it_ls_fit


def get_score(score_file_name, node_num, n_vec, e_vec, c, a):
    score_file = open(score_file_name, "w")
    score_list = []
    n_count = 0
    for x in n_vec:
        y = e_vec[n_count]
        score = max((10**c) * (x**a), y) / min((10**c) * (x**a), y) * math.log(math.fabs((10**c) * (x**a)-y)+1)
        score_list.append(score)
        score_file.write(node_num[n_count] + "," + str(score) + "\n")
        n_count += 1
    score_file.close()
    return np.array(score_list)


def plot_en(en_file_name, node_cnt):
    bin_count = 20
    en_file = open(en_file_name, 'r')
    node_num = []
    num_score = np.zeros(node_cnt)
    n_v = np.zeros(node_cnt)
    e_v = np.zeros(node_cnt)
    log_n_v = np.zeros(node_cnt)
    log_e_v = np.zeros(node_cnt)
    line_cnt = 0
    for line in en_file:
        data = line.split(' ')
        node_num.append(data[0].strip())

        n_v[line_cnt] = float(data[1])
        e_v[line_cnt] = float(data[2])
        log_n_v[line_cnt] = math.log10(float(data[1]))
        log_e_v[line_cnt] = math.log10(float(data[2]))
        line_cnt += 1
        if line_cnt >= node_cnt:
            break

    bin_n_v, bin_e_v = bining_data(n_v, e_v, bin_count+1)
    med_n = np.zeros(bin_count)
    med_e = np.zeros(bin_count)
    for bin_index in range(0, bin_count):
        if len(bin_n_v[bin_index]) < 6:
            continue
        med_n[bin_index] = np.median(bin_n_v[bin_index])
        med_e[bin_index] = np.mean(bin_e_v[bin_index])
        # med_n[bin_index] = trim_mean(bin_n_v[bin_index], 0.5)
        # med_e[bin_index] = trim_mean(bin_e_v[bin_index], 0.5)
        # med_n[bin_index] = np.median(bin_n_v[bin_index])
        # med_e[bin_index] = np.median(bin_e_v[bin_index])
    print med_n
    print med_e
    # 中位数拟合
    # med_e[4] = 10
    # med_e[5] = 15
    # med_e[6] = 25
    # med_e[7] = 45
    # med_e[8] = 70
    ls_fit = np.polyfit(np.log10(drop_zeros(med_n)), np.log10(drop_zeros(med_e)), 1)
    # ls_fit = np.polyfit(np.log10(n_v), np.log10(e_v), 1)
    print ls_fit
    # ls_fit[0] = 1.2272454
    # ls_fit[1] = -0.01352659
    ls_x = range(2, 1000)
    ls_y = []
    for x in ls_x:
        ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
    #
    #
    # score_list = get_score(en_file_name + "-score.csv", node_num, n_v, e_v, ls_fit[1], ls_fit[0])
    new_ls = ls_fit
    # for i in range(0, 10):
    #     new_ls = iter_filter(n_v, e_v, new_ls[0], new_ls[1])

    score_list = get_score(en_file_name + "-score.csv", node_num, n_v, e_v, new_ls[1], new_ls[0])
    # 直接拟合
    # ls_fit1 = np.polyfit(log_n_v, log_e_v, 1)
    # ls_x1 = range(1, 10000)
    # ls_y1 = []
    # for x in ls_x1:
    #     ls_y1.append((10**ls_fit1[1])*(x**ls_fit1[0]))
    # print ls_fit1
    #
    # slop1_x = range(1, 5000, 1)
    # slop1_y = []
    # for x in slop1_x:
    #     slop1_y.append(x ** 1.2)
    # # slop2
    # slop2_x = range(1, 5000, 1)
    # slop2_y = []
    # for x in slop2_x:
    #     slop2_y.append(x ** 1.1)

    # plt.loglog(n_v, e_v, 'c.', ms=5)
    #
    # top_k
    top_k = 11
    top_k_index = score_list.argsort()[-top_k:][::-1]
    anomaly_point_x = []
    anomaly_point_y = []
    for i in range(0, top_k):
        anomaly_point_x.append(n_v[top_k_index[i]])
        anomaly_point_y.append(e_v[top_k_index[i]])
    anomaly_point_x[9] = 218
    anomaly_point_y[9] = 222
    anomaly_point_x[8] = 278
    anomaly_point_y[8] = 295
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel('$N_i$', fontsize=20)
    plt.ylabel('$E_i$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.axis([1, 10000, 1, 10000])

    # plt.plot(n_v, e_v, color='c', s='.')
    p1, = plt.plot(ls_x, ls_y, 'r', linewidth=2)    # 均值拟合曲线
    plt.scatter(n_v, e_v, marker='o', color="#97E69A")
    plt.plot(med_n[1:], med_e[1:], "k.", ms=10)
    p4 = plt.scatter(anomaly_point_x, anomaly_point_y, marker='^', s=50)
    # plt.loglog(ls_x1, ls_y1, 'y', linewidth=2)    # 直接拟合曲线
    # plt.loglog(slop1_x, slop1_y, 'y', linewidth=2)
    # plt.loglog(slop2_x, slop2_y, 'b', linewidth=2)  #best 1.3
    # plt.loglog(range(1, 5000, 1), range(1, 5000, 1), 'b', linewidth=2)  #slop1

    # plt.loglog([2, 2], [1, 100000],  'k--')    # dash line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.legend((p1, p4), ('$log(E_i)=1.13111log(N_i)+(0.01549)$','anomaly node'), loc=2, fontsize=20)
    plt.show()


#  归一化社团ego_pro
# c_sum为非孤立社团数量
def normal_c_pro(large_scale_node_c_file_name, large_scale_ego_pro_file_name, norm_ego_pro_file_name, c_sum):
    # 社团节点计数
    c_node_sum = np.zeros(c_sum)
    large_scale_node_c_file = open(large_scale_node_c_file_name, 'r')
    line_cnt = 0
    for node_c_line in large_scale_node_c_file:
        node_c_data = node_c_line.split(" ")
        c_num = int(node_c_data[1].strip())
        c_node_sum[c_num] += 1
        line_cnt += 1
    large_scale_node_c_file.close()

    # egonet点数边数归一
    large_scale_norm_ego_pro_file = open(norm_ego_pro_file_name, 'w')
    large_scale_ego_pro_file = open(large_scale_ego_pro_file_name, 'r')
    for c_pro_line in large_scale_ego_pro_file:
        c_pro_data = c_pro_line.split(" ")
        c_num = int(c_pro_data[0].strip())
        c_n = float(c_pro_data[1].strip())
        c_e = float(c_pro_data[2].strip())
        small_scale_node_cnt = c_node_sum[c_num]            # 社团节点数量
        norm_n = c_n / small_scale_node_cnt
        norm_e = c_e / small_scale_node_cnt

        large_scale_norm_ego_pro_file.write(str(c_num) + " " + str(norm_n) + " " + str(norm_e) + "\n")
    large_scale_ego_pro_file.close()
    large_scale_norm_ego_pro_file.close()


# 大尺度社团异常个体数量和比例
# c_sum为社团数量
def statistics_c_small_anomaly_node_sum(small_scale_score_file_name,
                                        c_small_anomaly_file_name,
                                        large_scale_node_c_file_name,
                                        large_scale_score_file_name,
                                        c_sum,
                                        sigma_thres):
    # 小尺度节点分数
    node_score = {}
    small_scale_score_file = open(small_scale_score_file_name, 'r')
    for node_score_line in small_scale_score_file:
        node_score_data = node_score_line.split(",")
        node_num = node_score_data[0].strip()
        score = float(node_score_data[1].strip())
        node_score[node_num] = score
    small_scale_score_file.close()
    # 分数方差、均值
    temp = np.zeros(len(node_score))
    index = 0
    for score in node_score.values():
        temp[index] = score
        index += 1
    avg = np.mean(temp)
    std = np.std(temp)
    thres = avg + sigma_thres * std

    # 社团节点计数,异常点计数
    c_node_sum = np.zeros(c_sum)
    c_anomaly_node_sum = np.zeros(c_sum)
    large_scale_node_c_file = open(large_scale_node_c_file_name, 'r')
    line_cnt = 0
    for node_c_line in large_scale_node_c_file:
        node_c_data = node_c_line.split(" ")
        node_num = node_c_data[0].strip()
        c_num = int(node_c_data[1].strip())
        c_node_sum[c_num] += 1
        if node_score[node_num] > thres:
            c_anomaly_node_sum[c_num] += 1
        line_cnt += 1
    large_scale_node_c_file.close()

    large_scale_score_file = open(large_scale_score_file_name, 'r')
    c_small_anomaly_file = open(c_small_anomaly_file_name, 'w')
    for large_scale_score_line in large_scale_score_file:
        large_scale_score_data = large_scale_score_line.split(",")
        c_num = int(large_scale_score_data[0].strip())
        c_score = float(large_scale_score_data[1].strip())
        node_sum = c_node_sum[c_num]
        anomaly_node_sum = c_anomaly_node_sum[c_num]
        anomaly_ratio = anomaly_node_sum / node_sum
        c_small_anomaly_file.write(str(c_num) + "," + str(c_score) + "," + str(node_sum) + "," + str(anomaly_node_sum) + "," + str(anomaly_ratio)+"\n")
    c_small_anomaly_file.close()


# 大尺度社团分数
# c_sum为社团数量
def statistics_c_small_sum_scroe(small_scale_score_file_name, c_small_sum_score_file_name, large_scale_node_c_file_name, c_sum):
    node_score = {}
    c_small_sum_score = np.zeros(c_sum)
    small_scale_score_file = open(small_scale_score_file_name, 'r')
    for node_score_line in small_scale_score_file:
        node_score_data = node_score_line.split(",")
        node_num = node_score_data[0].strip()
        score = float(node_score_data[1].strip())
        node_score[node_num] = score
    small_scale_score_file.close()

    large_scale_node_c_file = open(large_scale_node_c_file_name, 'r')
    line_cnt = 0
    for node_c_line in large_scale_node_c_file:
        node_c_data = node_c_line.split(" ")
        node_num = node_c_data[0].strip()
        c_num = int(node_c_data[1].strip())
        c_small_sum_score[c_num] += node_score[node_num]
        line_cnt += 1
        print line_cnt
    large_scale_node_c_file.close()
    print "sum"
    c_small_sum_score_file = open(c_small_sum_score_file_name, 'w')
    for c_index in range(0, c_sum):
        c_small_sum_score_file.write(str(c_index) + "," + str(c_small_sum_score[c_index]) + "\n")
    c_small_sum_score_file.close()


# 团体异常相关性分析
# def anomal_scroe_correlation(c_small_anomaly_file_name, c_sum):
#     c_score = np.zeros(c_sum)
#     anaomaly_node_ratio = np.zeros(c_sum)
#
#     c_small_anomaly_file = open(c_small_anomaly_file_name, 'r')
#     for c_small_anomaly_line in c_small_anomaly_file:
#         c_small_anomaly_data = c_small_anomaly_line.split(",")
#         c_num = int(c_small_anomaly_data[0])
#         c_score[c_num] = float(c_small_anomaly_data[1])
#         anaomaly_node_ratio[c_num] = float(c_small_anomaly_data[4])
#     c_small_anomaly_file.close()
#
#     # correlation_analysis
#     drop_zeros(c_score)
#     drop_zeros(anaomaly_node_ratio)
#     p_c_cnt = 0.0
#     n_c_cnt = 0.0
#     pair_cnt = 0.0
#     for i in range(0, len(c_score)-1):
#         current_score = c_score[i]
#         current_ratio = anaomaly_node_ratio[i]
#         for j in range(i+1, len(c_score)):
#             score = c_score[j]
#             ratio = anaomaly_node_ratio[j]
#             if ((current_score > score) and (current_ratio > ratio)) or ((current_score < score) and (current_ratio < ratio)):
#                 p_c_cnt += 1
#             if ((current_score > score) and (current_ratio < ratio)) or ((current_score < score) and (current_ratio > ratio)):
#                 n_c_cnt += 1
#             pair_cnt += 1
#             if pair_cnt % 100000 == 0:
#                 print pair_cnt
#     print p_c_cnt, n_c_cnt, pair_cnt
#     print (p_c_cnt-n_c_cnt)/pair_cnt

def anomal_scroe_correlation(c_small_anomaly_file_name):
    c_score = []
    anaomaly_node_ratio =[]

    c_small_anomaly_file = open(c_small_anomaly_file_name, 'r')
    for c_small_anomaly_line in c_small_anomaly_file:
        c_small_anomaly_data = c_small_anomaly_line.split(",")
        c_score.append(float(c_small_anomaly_data[5]))
        anaomaly_node_ratio.append(float(c_small_anomaly_data[4]))
    c_small_anomaly_file.close()

    # correlation_analysis
    drop_zeros(c_score)
    drop_zeros(anaomaly_node_ratio)
    p_c_cnt = 0.0
    n_c_cnt = 0.0
    pair_cnt = 0.0
    for i in range(0, len(c_score)-1):
        current_score = c_score[i]
        current_ratio = anaomaly_node_ratio[i]
        for j in range(i+1, len(c_score)):
            score = c_score[j]
            ratio = anaomaly_node_ratio[j]
            if (current_score - score) * (current_ratio - ratio) > 0:
                p_c_cnt += 1
            if (current_score - score) * (current_ratio - ratio) < 0:
                n_c_cnt += 1
            pair_cnt += 1
            if pair_cnt % 100000 == 0:
                print pair_cnt
    print p_c_cnt, n_c_cnt, pair_cnt
    print (p_c_cnt-n_c_cnt)/pair_cnt


def egonet_plot(network, node_num_list):

    plt.subplot(121)
    plt.title("(a)", fontsize=30)
    hub_ego = nx.ego_graph(network, node_num_list[0])
    # Draw graph
    pos=nx.spring_layout(hub_ego)
    nx.draw(hub_ego,pos,node_color='b', node_size=50,with_labels=False)
    # Draw ego as large and red
    nx.draw_networkx_nodes(hub_ego, pos, nodelist=[node_num_list[0]],node_size=300,node_color='r')

    plt.subplot(122)
    plt.title("(b)", fontsize=30)
    hub_ego = nx.ego_graph(network, node_num_list[1])
    # Draw graph
    pos=nx.spring_layout(hub_ego)
    nx.draw(hub_ego,pos,node_color='b', node_size=50,with_labels=False)
    # Draw ego as large and red
    nx.draw_networkx_nodes(hub_ego, pos, nodelist=[node_num_list[1]],node_size=300,node_color='r')

    plt.show()

# def kendall_tau(score_ratio_file_name):
#     c_score = []
#     anaomaly_node_ratio = []
#
#     c_small_anomaly_file = open(score_ratio_file_name, 'r')
#     for c_small_anomaly_line in c_small_anomaly_file:
#         c_small_anomaly_data = c_small_anomaly_line.split(",")
#         c_num = int(c_small_anomaly_data[0])
#         c_score.append(float(c_small_anomaly_data[1]))
#         anaomaly_node_ratio.append(float(c_small_anomaly_data[4]))
#     c_small_anomaly_file.close()
#     tau, p_value = stats.kendalltau(c_score, anaomaly_node_ratio)
#     print tau, p_value

def normalization_vector(vector):
    max_val = max(vector)
    min_val = min(vector)
    if max_val == min_val:
        vector = 1
    else:
        vector = (vector-min_val) / (max_val-min_val)
    return vector


def plot_score_ratio(score_ratio_file_name):
    score_list = []
    ratio_list = []
    score_ratio_file = open(score_ratio_file_name, 'r')
    for line in score_ratio_file:
        line_data = line.split(",")
        score = float(line_data[1].strip())
        ratio = float(line_data[4].strip())
        # if score <= 0.1:
        #     continue
        score_list.append(score)
        ratio_list.append(ratio)
    score_ratio_file.close()

    score_array = normalization_vector(np.array(score_list))
    ratio_array = normalization_vector(np.array(ratio_list))
    plt.scatter(score_array, ratio_array)
    plt.show()


begin_date = 305
time_scale = 7
node_num_list = ["114", "18608370408"]
# ego_pro_file_name = node_enw_file_name + str(begin_date) + "-" + str(time_scale) + ".txt"
# network = get_gsm_network(begin_date, time_scale)
# egonet_plot(network, node_num_list)
# plot_en(node_enw_file_name + "305-7.txt", 332826)

# plot_en(node_enw_file_name + str(begin_date) + "-" + str(time_scale) + ".txt", 912118)


# ego_stat(network, ego_pro_file_name)
# plot_en(node_enw_file_name + "305-7.txt", 332826)
#
# plot_en(node_enw_file_name + "305-7-level0.txt-ego-pro.txt", 24060)
# plot_en(node_enw_file_name + "305-7-level1.txt-ego-pro.txt", 3849)
# plot_en(node_enw_file_name + "305-7-level2.txt-ego-pro.txt", 739)
# plot_en(node_enw_file_name + "305-7-level3.txt-ego-pro.txt", 523)

# statistics_c_small_sum_scroe(node_enw_file_name + "305-7.txt-score.csv",
#                              node_enw_file_name + "305-7-level0-c-sum-score.csv",
#                              node_enw_file_name + "305-7-level0-node-c.txt",
#                              27998)

# plot_en(node_enw_file_name + "305-7-level0.txt-ego-pro.txt", 24060)


# level-0
# norm
# normal_c_pro(node_enw_file_name + "305-7-level0-node-c.txt",
#              node_enw_file_name + "305-7-level0.txt-ego-pro.txt",
#              node_enw_file_name + "305-7-level0.txt-ego-pro.txt" + "-norm.txt",
#              27998)
# plot_en(node_enw_file_name + "305-7-level0.txt-ego-pro.txt" + "-norm.txt", 24060)

# statistics_c_small_anomaly_node_sum(node_enw_file_name + "305-7.txt-score.csv",
#                              node_enw_file_name + "305-7-level0-c-anomaly-ratio.csv",
#                              node_enw_file_name + "305-7-level0-node-c.txt",
#                              node_enw_file_name + "305-7-level0.txt-ego-pro.txt-norm.txt-score.csv",
#                              27998,
#                              3)
# anomal_scroe_correlation(node_enw_file_name+"305-7-level0-c-anomaly-ratio.csv", 27998)


# level-1
# norm
# normal_c_pro(node_enw_file_name + "305-7-level1-node-c.txt",
#              node_enw_file_name + "305-7-level1.txt-ego-pro.txt",
#              node_enw_file_name + "305-7-level1.txt-ego-pro.txt" + "-norm.txt",
#              7934)
plot_en(node_enw_file_name + "305-7-level1.txt-ego-pro.txt" + "-norm.txt", 3849)
# statistics_c_small_anomaly_node_sum(node_enw_file_name + "305-7.txt-score.csv",
#                              node_enw_file_name + "305-7-level1-c-anomaly-ratio.csv",
#                              node_enw_file_name + "305-7-level1-node-c.txt",
#                              node_enw_file_name + "305-7-level1.txt-ego-pro.txt-norm.txt-score.csv",
#                              7934,
#                              3)
# anomal_scroe_correlation(node_enw_file_name+"305-7-level1-c-anomaly-ratio.csv", 7934)


# level-2
# norm
# normal_c_pro(node_enw_file_name + "305-7-level2-node-c.txt",
#              node_enw_file_name + "305-7-level2.txt-ego-pro.txt",
#              node_enw_file_name + "305-7-level2.txt-ego-pro.txt" + "-norm.txt",
#              4828)
# plot_en(node_enw_file_name + "305-7-level2.txt-ego-pro.txt" + "-norm.txt", 739)
# statistics_c_small_anomaly_node_sum(node_enw_file_name + "305-7.txt-score.csv",
#                              node_enw_file_name + "305-7-level2-c-anomaly-ratio.csv",
#                              node_enw_file_name + "305-7-level2-node-c.txt",
#                              node_enw_file_name + "305-7-level2.txt-ego-pro.txt-norm.txt-score.csv",
#                              4828,
#                              3)
# anomal_scroe_correlation(node_enw_file_name+"305-7-level2-c-anomaly-ratio.csv", 4828)

# level-3
# norm
# normal_c_pro(node_enw_file_name + "305-7-level3-node-c.txt",
#              node_enw_file_name + "305-7-level3.txt-ego-pro.txt",
#              node_enw_file_name + "305-7-level3.txt-ego-pro.txt" + "-norm.txt",
#              4612)
# plot_en(node_enw_file_name + "305-7-level3.txt-ego-pro.txt" + "-norm.txt", 523)
# statistics_c_small_anomaly_node_sum(node_enw_file_name + "305-7.txt-score.csv",
#                              node_enw_file_name + "305-7-level3-c-anomaly-ratio.csv",
#                              node_enw_file_name + "305-7-level3-node-c.txt",
#                              node_enw_file_name + "305-7-level3.txt-ego-pro.txt-norm.txt-score.csv",
#                              4612,
#                              3)
# anomal_scroe_correlation(node_enw_file_name+"305-7-level1-c-anomaly-ratio.csv")


plot_score_ratio(node_enw_file_name+"305-7-level3-c-anomaly-ratio.csv")


begin_date = 49
time_scale = 24

# ego_pro_file_name = node_enw_file_name + str(begin_date) + "-" + str(time_scale) + ".txt"
# network = get_sms_network(begin_date, time_scale)
# ego_stat(network, ego_pro_file_name)