# -*- coding: gb18030 -*-
__author__ = 'yu'

import networkx as nx
import MySQLdb
import math
import matplotlib.pyplot as plt
import numpy as np

current_path = "D:\\ComplexNetwork"
node_enw_file = current_path + "\\result\\ego_pro\\"


def best_result():
    c = 0.00226032
    a = 1.06836182
    x = 116
    y = 118
    result_file_name = node_enw_file + "305-7.txt" + "_result.txt"
    result_file = open(result_file_name, 'w')
    en_file = open(node_enw_file + "305-7.txt", 'r')
    for line in en_file:
        data = line.split(' ')
        x = float(data[1])
        y = float(data[2])
        xx = max(10**0.108 * x**1.3, y) / min(10**0.108 * x**1.3, y) * math.log(math.fabs(10**0.108 * x**1.3-y)+1)
        b = max((10**c) * (x**a), y) / min((10**c) * (x**a), y) * math.log(math.fabs((10**c) * (x**a)-y)+1)
        result_file.write(data[0] + " " + str(xx) + " " + str(b) + "\n")
    en_file.close()
    result_file.close()


def drop_zeros(a_list):
    return [i for i in a_list if i > 0]


def log_binning(x, bin_count=10):
    min_x = math.log10(min(x))
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
        if not network.has_edge(row[1], row[2]):
            network.add_edge(row[1], row[2], weight=1)
        else:
            network.add_edge(row[1], row[2], weight=1+network.get_edge_data(row[1], row[2])['weight'])
    select_cur.close()
    conn.commit()
    conn.close()
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


def plot_en(en_file_name, node_cnt):
    bin_count = 10
    en_file = open(en_file_name, 'r')
    n_v = np.zeros(node_cnt)
    e_v = np.zeros(node_cnt)
    log_n_v = np.zeros(node_cnt)
    log_e_v = np.zeros(node_cnt)
    line_cnt = 0
    for line in en_file:
        data = line.split(' ')
        if float(data[1]) <= 0:
            continue
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
        med_n[bin_index] = sum(bin_n_v[bin_index])/len(bin_n_v[bin_index])
        med_e[bin_index] = np.median(bin_e_v[bin_index])

    ls_fit1 = np.polyfit(log_n_v, log_e_v, 1)
    ls_x1 = range(1, 10000)
    ls_y1 = []
    for x in ls_x1:
        ls_y1.append((10**ls_fit1[1])*(x**ls_fit1[0]))
    print ls_fit1
    # 中位数拟合
    ls_fit = np.polyfit(np.log10(med_n), np.log10(med_e), 1)
    ls_x = range(1, 10000)
    ls_y = []
    for x in ls_x:
        ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
    print ls_fit
    slop1_x = range(1, 5000, 1)
    slop1_y = []
    for x in slop1_x:
        slop1_y.append(x ** 1.2)
    # # slop2
    # slop2_x = range(1, 5000, 1)
    # slop2_y = []
    # for x in slop2_x:
    #     slop2_y.append(x ** 1.1)

    plt.loglog(n_v, e_v, "go")
    plt.loglog(med_n, med_e, "ko", ms=10)
    plt.loglog(ls_x, ls_y, 'r', linewidth=2)    # 拟合曲线
    # plt.loglog(ls_x1, ls_y1, 'y', linewidth=2)    # 拟合曲线
    plt.loglog(slop1_x, slop1_y, 'y', linewidth=2)
    # plt.loglog(slop2_x, slop2_y, 'b', linewidth=2)  #best 1.3
    # plt.loglog(range(1, 5000, 1), range(1, 5000, 1), 'b', linewidth=2)  #slop1
    # plt.plot(log_n_v, log_e_v, "go")
    # plt.loglog([2, 2], [1, 100000],  'k--')    # 拟合曲线
    plt.show()

# def plot_en(en_file_name, node_cnt):
#     en_file = open(en_file_name, 'r')
#     n_v = []
#     e_v = []
#
#     for line in en_file:
#
#         data = line.split(' ')
#         n_v.append(float(data[1]))
#         e_v.append(float(data[2]))
#         if float(data[1]) > 100 and data[1] == data[2]:
#             print data[0]

begin_date = 305
time_scale = 7
ego_pro_file_name = node_enw_file + str(begin_date) + "-" + str(time_scale) + ".txt"
network = get_gsm_network(begin_date, time_scale)
ego_stat(network, ego_pro_file_name)
plot_en(node_enw_file + "305-7.txt", 332826)

# plot_en(node_enw_file + "1-1.txt", 104958)
# plot_en(node_enw_file + "1-7.txt", 288790)
# plot_en(node_enw_file + "1-28.txt", 578633)
# plot_en(node_enw_file + "305-7.txt", 332826)

# 305   2010/11/1
# ego_stat(305, 7)
# network = get_gsm_network(1, 28)
# ego_net = nx.ego_graph(network, "15808375004", radius=1)
# print 1

# x = 5985
# y = 7150
# c = 0.00226032
# a = 1.06836182
# # print math.log(6474), math.log10(100)
# print max(10**0.108 * x**1.3, y) / min(10**0.108 * x**1.3, y) * math.log(math.fabs(10**0.108 * x**1.3-y)+1), max(10**c * x**a, y) / min(10**c * x**a, y) * math.log(math.fabs(10**c * x**a-y)+1)



# [ 1.06836182  0.00226032]
# [ 1.01510372  0.06930474]
