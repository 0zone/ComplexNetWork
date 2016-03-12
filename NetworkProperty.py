# -*- coding: gb18030 -*-
__author__ = 'yu'

import networkx as nx
import MySQLdb

aba_gsm_path = "D:\\ComplexNetwork\\result\\network_pro\\gsm\\"
aba_sms_path = "D:\\ComplexNetwork\\result\\network_pro\\sms\\"
as_733_path = "D:\\ComplexNetwork\\result\\network_pro\\as\\"
aba_gsm_node_cnt = 4270930  # 1到530

aba_sms_node_cnt = 9330493
as_733_node_cnt = 7716


def get_gsm_network_property(time_scale):
    table_name = 'aba_gsm'
    min_date_index = 1
    max_date_index = 531
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index >= %s and date_index < %s'

    file_name = aba_gsm_path + str(time_scale) + ".txt"
    file = open(file_name, 'w')

    for begin_date_index in range(min_date_index, max_date_index, time_scale):
        end_date_index = begin_date_index + time_scale
        smash_date_index = begin_date_index / time_scale + 1
        print time_scale, begin_date_index

        select_cur = conn.cursor()
        select_cur.execute(sql_select, (begin_date_index, end_date_index))
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

        # 存储结果
        node_cnt = network.number_of_nodes()
        edge_cnt = network.number_of_edges()
        degree = float(sum(network.degree(weight='weight').values())) / aba_gsm_node_cnt
        average_clustering = (nx.average_clustering(network, weight='weight') * node_cnt) / aba_gsm_node_cnt
        number_connected_components = nx.number_connected_components(network)
        # density = (2 * edge_cnt) / aba_gsm_node_cnt*(aba_gsm_node_cnt-1)
        # degree_assortativity_coefficient = nx.degree_assortativity_coefficient(network, weight='weight')
        # print 'ass'
        file.write(str(smash_date_index) + '    ' + str(degree) + '    ' + str(average_clustering) + '    ' + str(number_connected_components) + '\n')
    file.close()
    conn.close()


def get_sms_network_property(time_scale):
    table_name = 'aba_sms'
    min_date_index = 49
    max_date_index = 745
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index >= %s and date_index < %s'

    file_name = aba_sms_path + str(time_scale) + ".txt"
    file = open(file_name, 'w')

    for begin_date_index in range(min_date_index, max_date_index, time_scale):
        end_date_index = begin_date_index + time_scale
        smash_date_index = (begin_date_index-min_date_index) / time_scale + 1
        print time_scale, begin_date_index

        select_cur = conn.cursor()
        select_cur.execute(sql_select, (begin_date_index, end_date_index))
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

        node_cnt = 0
        edge_cnt = 0
        degree = 0
        average_clustering = 0
        number_connected_components = 0
        if len(result_data) > 0:
            # 存储结果
            node_cnt = network.number_of_nodes()
            edge_cnt = network.number_of_edges()
            degree = float(sum(network.degree(weight='weight').values())) / aba_sms_node_cnt
            average_clustering = (nx.average_clustering(network, weight='weight') * node_cnt) / aba_sms_node_cnt
            number_connected_components = nx.number_connected_components(network)

        file.write(str(smash_date_index) + '    ' + str(node_cnt) + '    ' + str(edge_cnt) + '    ' + str(degree) + '    ' + str(average_clustering) + '    ' + str(number_connected_components) + '\n')
    file.close()
    conn.close()


def get_as_network_property(time_scale):
    table_name = 'as733'
    min_date_index = 1
    max_date_index = 734
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index >= %s and date_index < %s'

    file_name = as_733_path + str(time_scale) + ".txt"
    file = open(file_name, 'w')

    for begin_date_index in range(min_date_index, max_date_index, time_scale):
        end_date_index = begin_date_index + time_scale
        smash_date_index = (begin_date_index-min_date_index) / time_scale + 1
        print time_scale, begin_date_index

        select_cur = conn.cursor()
        select_cur.execute(sql_select, (begin_date_index, end_date_index))
        result_data = select_cur.fetchall()
        # 构建网络

        network = nx.Graph()
        for row in result_data:
            network.add_edge(row[1], row[2])
            # if not network.has_edge(row[1], row[2]):
            #     network.add_edge(row[1], row[2], weight=1)
            # else:
            #     network.add_edge(row[1], row[2], weight=1+network.get_edge_data(row[1], row[2])['weight'])
        select_cur.close()
        conn.commit()

        node_cnt = 0
        edge_cnt = 0
        degree = 0
        average_clustering = 0
        number_connected_components = 0
        if len(result_data) > 0:
            # 存储结果
            node_cnt = network.number_of_nodes()
            edge_cnt = network.number_of_edges()
            degree = float(sum(network.degree().values())) / as_733_node_cnt
            average_clustering = (nx.average_clustering(network) * node_cnt) / as_733_node_cnt
            number_connected_components = nx.number_connected_components(network)

        file.write(str(smash_date_index) + '    ' + str(node_cnt) + '    ' + str(edge_cnt) + '    ' + str(degree) + '    ' + str(average_clustering) + '    ' + str(number_connected_components) + '\n')
    file.close()
    conn.close()


# 40 49 50 60 70
# get_gsm_network_property(70)
# get_sms_network_property(72)
get_sms_network_property(50)
get_sms_network_property(54)
get_sms_network_property(60)
# for time_scale in range(3, 70, 2):
#     get_sms_network_property(time_scale)

# for time_scale in range(70, 150):
#     get_as_network_property(time_scale)