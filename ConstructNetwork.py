# -*- coding: gb18030 -*-
__author__ = 'yu'


import time
import networkx as nx
import matplotlib.pyplot as plt
import MySQLdb

time_format = time_format = '%m-%d-%H-%M'
result_file_path = 'result\\' + time.strftime(time_format, time.localtime()) + '-'

# table_name = 'as733'
as_max_date_index = 734                        # 最大时间
as_node_num_all = 7716                         # 点数

plot_style = ['ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点
as_window_size_arr = [1, 7, 49, 70]            # 窗口
aba_sms_window_size_arr = [3600, 86400, 259200, 604800]
# plot_style = ['r', 'b', 'c', 'g', 'm', 'y', 'k']            # 线


# 通过date_index获取相关数据
def get_data_by_date_index(date_index, table_name):
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
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


# 构建aba_sms m-smash网络
def get_aba_sms_m_smash_network(begin_timestamp, m_smash):
    table_name = 'aba_sms'
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE time_stamp >= %s and time_stamp < %s'
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    cur = conn.cursor()

    smash_network = nx.Graph()

    end_timestamp = begin_timestamp + m_smash
    cur.execute(sql_select, (begin_timestamp, end_timestamp))
    result_data = cur.fetchall()
    for row in result_data:
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
    cluster_cof = {}

    for i in range(0, len(size_arr)):            # 窗口
        date_index[i] = []
        node_num[i] = []
        edge_num[i] = []
        self_loop_num[i] = []
        avg_degree[i] = []
        avg_degree_all[i] = []
        density[i] = []
        cluster_cof[i] = []
        window_size = size_arr[i]                # 窗口大小
        for j in range(1, as_max_date_index, window_size):
            if as_max_date_index - j < window_size:
                break
            network = get_as_m_smash_network(j, window_size, as_max_date_index)

            date_index[i].append(j)
            node_num[i].append(network.number_of_nodes())
            edge_num[i].append(network.number_of_edges())
            self_loop_num[i].append(network.number_of_selfloops())
            avg_degree[i].append(float(sum(network.degree().values()))/network.number_of_nodes())     # 平均度
            # avg_degree_all[i].append(float(sum(network.degree().values()))/7716)
            density[i].append(nx.density(network))
            cluster_cof[i].append(nx.clustering(network))
            

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

    # # avg_degree_all
    # for i in range(0, len(size_arr)):           # 窗口
    #     plt.plot(date_index[i], avg_degree_all[i], plot_style[i], label=size_arr[i])
    # plt.xlabel('time')
    # plt.ylabel('avg degree(all)')
    # plt.legend()
    # plt.savefig(result_file_path + 'avg_degree_all.pdf')
    # plt.show()

    # density
    for i in range(0, len(size_arr)):           # 窗口
        plt.plot(date_index[i], density[i], plot_style[i], label=size_arr[i])
    plt.xlabel('time')
    plt.ylabel('density')
    plt.legend()
    plt.savefig(result_file_path + 'density.pdf')
    plt.show()


def get_sms_network_node_property(time_scale):
    table_name = 'aba_sms'
    min_date_index = 1
    max_date_index = 744
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index >= %s and date_index < %s'
    sql_insert = 'INSERT INTO ' + table_name + '_' + str(time_scale) + 'h' + ' (num, out_degree, in_degree, clu, date_index) VALUES(%s, %s, %s,%s,%s);'

    for begin_date_index in range(min_date_index, max_date_index, time_scale):
        end_date_index = begin_date_index + time_scale
        smash_date_index = begin_date_index / time_scale + 1
        print smash_date_index

        select_cur = conn.cursor()
        select_cur.execute(sql_select, (begin_date_index, end_date_index))
        result_data = select_cur.fetchall()
        # 构建网络
        out_degree = {}
        in_degree = {}
        network = nx.Graph()
        for row in result_data:
            # network.add_edge(row[0], row[1], weight=1)
            if row[1] in out_degree:
                out_degree[row[1]] += 1
            else:
                out_degree[row[1]] = 1
            if row[2] in in_degree:
                in_degree[row[2]] += 1
            else:
                in_degree[row[2]] = 1

            if not network.has_edge(row[1], row[2]):
                network.add_edge(row[1], row[2], weight=1)
            else:
                network.add_edge(row[1], row[2], weight=1+network.get_edge_data(row[1], row[2])['weight'])
        select_cur.close()
        conn.commit()
        clu = nx.clustering(network)
        # number_connected_components
        # average_clustering
        # rich_club_coefficient
        # pagerank      点
        rows = []
        for node in network.nodes_iter():
            out_value = 0
            in_value = 0
            if node in out_degree:
                out_value = out_degree[node]
            if node in in_degree:
                in_value = in_degree[node]
            rows.append((node, out_value, in_value, clu[node], smash_date_index))

        insert_cur = conn.cursor()
        insert_cur.executemany(sql_insert, rows)
        conn.commit()
        print 'row'
        # in_degree = network.in_degree(weight='weight')
        # out_degree = network.out_degree(weight='weight')
        # un_di_network = nx.Graph()
        # for n, nbrs in network.adjacency_iter():
        #     for nbr, edict in nbrs.items():
        #         sum_value = sum([d['weight'] for d in edict.values()])
        #         if not un_di_network.has_edge(n, nbr):
        #             un_di_network.add_edge(n, nbr, weight=sum_value)
        #         else:
        #             un_di_network.add_edge(n, nbr, weight=sum_value + un_di_network.get_edge_data(n, nbr)['weight'])
        # clu = nx.clustering(un_di_network)
        # rows = []
        # for node in un_di_network.nodes_iter():
        #     rows.append((node, out_degree[node], in_degree[node], clu[node], smash_date_index))
        #
        # insert_cur = conn.cursor()
        # insert_cur.executemany(sql_insert, rows)
        # conn.commit()

    conn.close()

        # # print network.number_of_selfloops()
        # print nx.density(network)
        # print nx.average_clustering(network)
        #
        # print float(sum(network.degree().values()))/network.number_of_nodes()     # 平均度
        # # avg_degree_all[i].append(float(sum(network.degree().values()))/7716)


def get_gsm_network_node_property(time_scale):
    table_name = 'aba_gsm'
    min_date_index = 1
    max_date_index = 730
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index >= %s and date_index < %s'
    sql_insert = 'INSERT INTO ' + table_name + '_' + str(time_scale) + 'd' + ' (num, out_degree, in_degree, clu, date_index) VALUES(%s, %s, %s,%s,%s);'

    for begin_date_index in range(min_date_index, max_date_index, time_scale):
        end_date_index = begin_date_index + time_scale
        smash_date_index = begin_date_index / time_scale + 1
        print smash_date_index

        select_cur = conn.cursor()
        select_cur.execute(sql_select, (begin_date_index, end_date_index))
        result_data = select_cur.fetchall()
        # 构建网络
        out_degree = {}
        in_degree = {}
        network = nx.Graph()
        for row in result_data:
            # network.add_edge(row[0], row[1], weight=1)
            if row[1] in out_degree:
                out_degree[row[1]] += 1
            else:
                out_degree[row[1]] = 1
            if row[2] in in_degree:
                in_degree[row[2]] += 1
            else:
                in_degree[row[2]] = 1

            if not network.has_edge(row[1], row[2]):
                network.add_edge(row[1], row[2], weight=1)
            else:
                network.add_edge(row[1], row[2], weight=1+network.get_edge_data(row[1], row[2])['weight'])
        select_cur.close()
        conn.commit()
        clu = nx.clustering(network)
        rows = []
        for node in network.nodes_iter():
            out_value = 0
            in_value = 0
            if node in out_degree:
                out_value = out_degree[node]
            if node in in_degree:
                in_value = in_degree[node]
            rows.append((node, out_value, in_value, clu[node], smash_date_index))

        insert_cur = conn.cursor()
        insert_cur.executemany(sql_insert, rows)
        conn.commit()

    conn.close()

get_gsm_network_node_property(30)
# get_sms_network_node_property(24)
# analyse_network_property(as_window_size_arr)

# g = get_m_smash_network(1, 1, max_date_index)
# print nx.average_clustering(g) * g.number_of_nodes() / 7716
# print float(sum(g.degree().values()))/7716










