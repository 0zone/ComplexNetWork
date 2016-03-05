# -*- coding: gb18030 -*-
__author__ = 'yu'

import networkx as nx
import MySQLdb

aba_gsm_path = "D:\\ComplexNetwork\\result\\network_pro\\"
aba_gsm_node_cnt = 4270930  # 1到530


def get_gsm_network_property(time_scale):
    table_name = 'aba_gsm'
    min_date_index = 1
    max_date_index = 530
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

# get_gsm_network_property(70)

for time_scale in range(14, 30):
    print time_scale
    get_gsm_network_property(time_scale)