# -*- coding: gb18030 -*-
__author__ = 'yu'
from community import *
import networkx as nx
import matplotlib.pyplot as plt
import MySQLdb
from Egonet import ego_stat, get_gsm_network

current_path = "D:/ComplexNetwork"
node_enw_file = current_path + "/result/ego_pro/"


begin_date = 305
time_scale = 7
ego_pro_file_name = node_enw_file + str(begin_date) + "-" + str(time_scale) + ".txt"
network = get_gsm_network(begin_date, time_scale)

dendo = generate_dendogram(network)
for level in range(len(dendo) - 1):
    partition = partition_at_level(dendo, level)
    comm_set = set()
    for k, v in partition.items():
        comm_set.add(v)
    print "partition at level", level, "is", len(comm_set)
    # level子图   粗粒化
    level_ind = induced_graph(partition, network)
    # 去除自环

    ego_stat(level_ind, "1")

# ind = induced_graph(part, g)