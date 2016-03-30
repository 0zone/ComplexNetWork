# -*- coding: gb18030 -*-
__author__ = 'yu'
from community import *
import networkx as nx
import matplotlib.pyplot as plt
import MySQLdb
from Egonet import ego_stat, get_gsm_network, get_sms_network

current_path = "D:/ComplexNetwork"
node_en_file_name = current_path + "/result/ego_pro/sms/"


begin_date = 305
time_scale = 7
ego_pro_file_name = node_en_file_name + str(begin_date) + "-" + str(time_scale) + ".txt"
# network = get_gsm_network(begin_date, time_scale)

begin_date = 49
time_scale = 24
network = get_sms_network(begin_date, time_scale)
print network.number_of_edges()
print network.number_of_nodes()
# dendo = generate_dendogram(network)
# for level in range(len(dendo) - 1):
#     partition_en_file_name = node_en_file_name + str(begin_date) + "-" + str(time_scale) + "-level" + str(level) + ".txt"
#     partition_node_c_file_name = node_en_file_name + str(begin_date) + "-" + str(time_scale) + "-level" + str(level) + "-node-c.txt"
#     partition_en_file = open(partition_en_file_name, 'w')
#     partition_node_c_file = open(partition_node_c_file_name, 'w')
#     print "level" + str(level)
#     partition = partition_at_level(dendo, level)
#     # �ڵ��������
#     for node_num, c_index in partition.items():
#         partition_node_c_file.write(str(node_num) + " " + str(c_index) + "\n")
#     # level��ͼ   ������     # ��ͼ�ṹ
#     level_ind = induced_graph(partition, network)
#     for edge in level_ind.edges():
#         partition_en_file.write(str(edge[0]) + " " + str(edge[1]) + "\n")
#
#     partition_node_c_file.close()
#     partition_en_file.close()

# ind = induced_graph(part, g)