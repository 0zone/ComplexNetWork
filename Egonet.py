# -*- coding: gb18030 -*-
__author__ = 'yu'

import networkx as nx
import MySQLdb
import math
import matplotlib.pyplot as plt
import numpy as np

current_path = "D:\\ComplexNetwork"
node_enw_file = current_path + "\\result\\ego_pro\\"


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


# def ego_stat(begin_date, time_scale):
#     network = get_gsm_network(begin_date, time_scale)
#     ego_pro_file_name = node_enw_file + str(begin_date) + "-" + str(time_scale) + ".txt"
#     ego_pro_file = open(ego_pro_file_name, 'w')
#     for node in network.nodes_iter():
#         neighbors_list = network.neighbors(node)
#         neighbors_list.append(node)
#         ego_net = network.subgraph(neighbors_list)
#         n = len(neighbors_list) - 1
#         e = ego_net.number_of_edges()
#         w = sum(ego_net.degree(weight='weight').values()) / 2
#         result = []
#         result.append(node)
#         result.append(str(n))
#         result.append(str(e))
#         result.append(str(w))
#         result.append('\n')
#         ego_pro_file.write(' '.join(result))
#     ego_pro_file.close()


def ego_stat(begin_date, time_scale):
    network = get_gsm_network(begin_date, time_scale)
    ego_pro_file_name = node_enw_file + str(begin_date) + "-" + str(time_scale) + ".txt"
    ego_pro_file = open(ego_pro_file_name, 'w')
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
        log_n_v[line_cnt] = math.log(float(data[1]))
        log_e_v[line_cnt] = math.log(float(data[2]))
        line_cnt += 1
        if line_cnt >= node_cnt:
            break

    ls_fit = np.polyfit(log_n_v, log_e_v, 1)
    poly = np.poly1d(ls_fit)
    print poly
    ls_x = range(1, 10000)
    ls_y = poly(ls_x)
    # 残差    0.006443  0.128510233671  0.391973701013      3 2.4658900091273404299221790935969     4 3.2661861635294543997750412981632
    h_e = poly(log_n_v)
    re = h_e-log_e_v
    a = (sum(re**2)/node_cnt)**0.5



    slop1_x = range(1, 5000, 1)
    slop1_y = []
    for x in slop1_x:
        slop1_y.append(x ** 1.3)
    # slop2
    slop2_x = range(1, 5000, 1)
    slop2_y = []
    for x in slop2_x:
        slop2_y.append(x ** 1.1)

    # for i in len(n_v):
    #     print max(10**0.108 * n_v[i]**1.3, e_v[i]) / min(10**0.108 * n_v[i]**1.3, e_v[i]) * math.log(10**0.108 * n_v[i]**1.3 - e_v[i]+1)

    # plt.loglog(n_v, e_v, "go")
    # plt.loglog(ls_x, ls_y, 'r', linewidth=2)    # 拟合曲线
    # plt.loglog(slop1_x, slop1_y, 'y', linewidth=2)
    # plt.loglog(slop2_x, slop2_y, 'b', linewidth=2)
    # plt.loglog(range(1, 5000, 1), range(1, 5000, 1), 'b', linewidth=2)
    plt.plot(log_n_v, log_e_v, "go")
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


# plot_en(node_enw_file + "1-1.txt", 104958)
# plot_en(node_enw_file + "1-7.txt", 288790)
plot_en(node_enw_file + "1-28.txt", 578633)

# 305   2010/11/1
# ego_stat(305, 7)
# network = get_gsm_network(1, 28)
# ego_net = nx.ego_graph(network, "15808375004", radius=1)
# print 1
x = 27
y = 93

print (10**0.108) * (x**1.3)
print max(10**0.108 * x**1.3, y) / min(10**0.108 * x**1.3, y) *math.log(math.fabs(10**0.108 * x**1.3- y)+1)

