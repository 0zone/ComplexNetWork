# -*- coding: gb18030 -*-
__author__ = 'yu'


import time
import networkx as nx
import matplotlib.pyplot as plt
import MySQLdb

time_format = time_format = '%m-%d-%H-%M'
result_file_path = 'result\\' + time.strftime(time_format, time.localtime()) + '-'

# table_name = 'as733'
max_date_index = 734                        # 最大时间
node_num_all = 7716                         # 点数

# window_size_arr = [1, 7, 49, 70]            # 窗口
# plot_style = ['ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点
window_size_arr = [1, 3, 5, 7, 9, 11, 15]            # 窗口
plot_style = ['r', 'b', 'c', 'g', 'm', 'y', 'k']            # 线


# 通过date_index获取相关数据
def get_data_by_date_index(date_index, table_name):
    conn = MySQLdb.connect(host="localhost", user="root", passwd="o.lyw%Rff1p;", db="network", charset="utf8")
    cur = conn.cursor()

    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index = %s'
    cur.execute(sql_select, date_index)
    result_data = cur.fetchall()

    cur.close()
    conn.commit()
    conn.close()

    return result_data


# 构建AS-733 m-smash网络
def get_as_m_smash_network(begin_date_index, m_smash, max_date):
    table_name = 'as733'
    end_date_index = min(begin_date_index + m_smash, max_date)
    smash_network = nx.Graph()
    for date_index in range(begin_date_index, end_date_index):
        result = get_data_by_date_index(date_index, table_name)
        for row in result:
            # 加入网络
            # if row[1] == row[2]:  # 自环
            #     continue
            smash_network.add_edge(row[1], row[2])

    return smash_network


# 网络性质分析
def analyse_network_property(size_arr):
    date_index = {}
    node_num = {}
    edge_num = {}
    self_loop_num = {}
    avg_degree = {}
    avg_degree_all = {}
    density = {}

    for i in range(0, len(size_arr)):            # 窗口
        date_index[i] = []
        node_num[i] = []
        edge_num[i] = []
        self_loop_num[i] = []
        avg_degree[i] = []
        avg_degree_all[i] = []
        density[i] = []
        window_size = size_arr[i]                # 窗口大小
        for j in range(1, max_date_index, window_size):
            if max_date_index - j < window_size:
                break
            network = get_as_m_smash_network(j, window_size, max_date_index)

            date_index[i].append(j)
            node_num[i].append(network.number_of_nodes())
            edge_num[i].append(network.number_of_edges())
            self_loop_num[i].append(network.number_of_selfloops())
            avg_degree[i].append(float(sum(network.degree().values()))/network.number_of_nodes())     # 平均度
            avg_degree_all[i].append(float(sum(network.degree().values()))/7716)
            density[i].append(nx.density(network))

    # node_num
    for i in range(0, len(size_arr)):           # 窗口
        plt.plot(date_index[i], node_num[i], plot_style[i], label=size_arr[i])
    plt.xlabel('time')
    plt.ylabel('number of nodes')
    plt.legend()
    plt.savefig(result_file_path + 'node_num.pdf')
    plt.show()

    # edge_num
    for i in range(0, len(size_arr)):           # 窗口
        plt.plot(date_index[i], edge_num[i], plot_style[i], label=size_arr[i])
    plt.xlabel('time')
    plt.ylabel('number of edges')
    plt.legend()
    plt.savefig(result_file_path + 'edge_num.pdf')
    plt.show()

    # self_loop_num
    for i in range(0, len(size_arr)):           # 窗口
        plt.plot(date_index[i], self_loop_num[i], plot_style[i], label=size_arr[i])
    plt.xlabel('time')
    plt.ylabel('number of self_loop')
    plt.legend()
    plt.savefig(result_file_path + 'self_loop_num.pdf')
    plt.show()

    # avg_degree
    for i in range(0, len(size_arr)):           # 窗口
        plt.plot(date_index[i], avg_degree[i], plot_style[i], label=size_arr[i])
    plt.xlabel('time')
    plt.ylabel('avg degree')
    plt.legend()
    plt.savefig(result_file_path + 'avg_degree.pdf')
    plt.show()

    # avg_degree_all
    for i in range(0, len(size_arr)):           # 窗口
        plt.plot(date_index[i], avg_degree_all[i], plot_style[i], label=size_arr[i])
    plt.xlabel('time')
    plt.ylabel('avg degree(all)')
    plt.legend()
    plt.savefig(result_file_path + 'avg_degree_all.pdf')
    plt.show()

    # density
    for i in range(0, len(size_arr)):           # 窗口
        plt.plot(date_index[i], density[i], plot_style[i], label=size_arr[i])
    plt.xlabel('time')
    plt.ylabel('density')
    plt.legend()
    plt.savefig(result_file_path + 'density.pdf')
    plt.show()


analyse_network_property(window_size_arr)

# g = get_m_smash_network(1, 1, max_date_index)
# print nx.average_clustering(g) * g.number_of_nodes() / 7716
# print float(sum(g.degree().values()))/7716










