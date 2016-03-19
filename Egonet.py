# -*- coding: gb18030 -*-
__author__ = 'yu'

import networkx as nx
import MySQLdb
import math
import matplotlib.pyplot as plt
import numpy as np

current_path = "D:\\ComplexNetwork"
node_enw_file_name = current_path + "\\result\\ego_pro\\"


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


def plot_en(en_file_name, node_cnt):
    bin_count = 10
    en_file = open(en_file_name, 'r')
    node_num_dict = {}
    num_score = np.zeros(node_cnt)
    n_v = np.zeros(node_cnt)
    e_v = np.zeros(node_cnt)
    log_n_v = np.zeros(node_cnt)
    log_e_v = np.zeros(node_cnt)
    line_cnt = 0
    for line in en_file:
        data = line.split(' ')
        node_num_dict[data[0]] = line_cnt

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
        if len(bin_n_v[bin_index]) < 10:
            continue
        med_n[bin_index] = np.mean(bin_n_v[bin_index])
        med_e[bin_index] = np.mean(bin_e_v[bin_index])
        # med_n[bin_index] = trim_mean(bin_n_v[bin_index], 0.2)
        # med_e[bin_index] = trim_mean(bin_e_v[bin_index], 0.2)
        # med_n[bin_index] = np.median(bin_n_v[bin_index])
        # med_e[bin_index] = np.median(bin_e_v[bin_index])
    print med_n
    print med_e
    # 中位数拟合
    ls_fit = np.polyfit(np.log10(med_n[0:6]), np.log10(med_e[0:6]), 1)
    ls_x = range(1, 10000)
    ls_y = []
    for x in ls_x:
        ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
    print ls_fit
    # new_ls = ls_fit
    # for i in range(0, 10):
    #     new_ls = iter_filter(n_v, e_v, new_ls[0], new_ls[1])

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
    plt.loglog(med_n, med_e, "k.", ms=10)
    plt.loglog(ls_x, ls_y, 'r', linewidth=2)    # 均值拟合曲线
    # plt.loglog(ls_x1, ls_y1, 'y', linewidth=2)    # 直接拟合曲线
    # plt.loglog(slop1_x, slop1_y, 'y', linewidth=2)
    # plt.loglog(slop2_x, slop2_y, 'b', linewidth=2)  #best 1.3
    # plt.loglog(range(1, 5000, 1), range(1, 5000, 1), 'b', linewidth=2)  #slop1
    plt.plot(n_v, e_v, "c.")
    plt.loglog([2, 2], [1, 100000],  'k--')    # dash line
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
# ego_pro_file_name = node_enw_file_name + str(begin_date) + "-" + str(time_scale) + ".txt"
# network = get_gsm_network(begin_date, time_scale)
# ego_stat(network, ego_pro_file_name)
# plot_en(node_enw_file_name + "305-7.txt", 332826)

network_file_name = node_enw_file_name + "305-7-level0.txt"
network = get_network_from_file(network_file_name)
ego_stat(network, network_file_name + "-ego-pro.txt")

network_file_name = node_enw_file_name + "305-7-level1.txt"
network = get_network_from_file(network_file_name)
ego_stat(network, network_file_name + "-ego-pro.txt")

network_file_name = node_enw_file_name + "305-7-level2.txt"
network = get_network_from_file(network_file_name)
ego_stat(network, network_file_name + "-ego-pro.txt")

network_file_name = node_enw_file_name + "305-7-level3.txt"
network = get_network_from_file(network_file_name)
ego_stat(network, network_file_name + "-ego-pro.txt")

