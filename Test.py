# -*- coding: gb18030 -*-
__author__ = 'yu'

import time
import networkx as nx                   #导入networkx包
import matplotlib.pyplot as plt     #导入绘图包matplotlib（需要安装，方法见第一篇笔记）
import MySQLdb

# GG = nx.Graph()
# MG=nx.MultiGraph()
# MG.add_weighted_edges_from([(130865333321308000001,2,.5), (1,2,.75), (2,3,.9), (1,4,0.5), (4,3,0.2)])
# print MG.degree(weight='weight')
# print MG.edges()
G = nx.MultiDiGraph()

G.add_weighted_edges_from([ (1,2,4), (1,2,1), (2,3,2),(3,2,1) ,(1,4,2), (4,3,1), (1,3,1), (2,1,1)])
# print G.degree()
in_degree = G.in_degree(weight='weight')
out_degree = G.out_degree(weight='weight')

GG = nx.Graph()
for n, nbrs in G.adjacency_iter():
    for nbr, edict in nbrs.items():
        sum_value = sum([d['weight'] for d in edict.values()])
        if not GG.has_edge(n, nbr):
            GG.add_edge(n, nbr, weight=sum_value)
        else:
            GG.add_edge(n, nbr, weight=sum_value + GG.get_edge_data(n, nbr)['weight'])

print GG.get_edge_data(3,2)

clu = nx.clustering(GG)
# print clu
# print nx.pagerank(GG)
# print nx.number_connected_components(GG)
# print nx.average_clustering(GG)
print nx.rich_club_coefficient(GG)

dic = {}

# row = []
# for node in GG.nodes_iter():
#     row.append((node, out_degree[node], in_degree[node], clu[node], 1))
# table_name = 'aba_sms'
# time_scale = 24
# conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
# sql_insert = 'INSERT INTO ' + table_name + '_' + str(time_scale) + 'h' + ' (num, out_degree, in_degree, clu, date_index) VALUES(%s, %s, %s,%s,%s);'
#
# insert_cur = conn.cursor()
# insert_cur.executemany(sql_insert, row)
# conn.commit()
# for edge in G.edges_iter():
#     print edge
#     # print G.get_edge_data(edge[0], edge[1])
#     for k in G.get_edge_data(edge[0], edge[1]):
#         print G.get_edge_data(edge[0], edge[1])[0]
# print nx.clustering(G, weight='weight')

# for n,nbrs in MG.adjacency_iter():
#     for nbr,edict in nbrs.items():
#         minvalue=sum([d['weight'] for d in edict.values()])
#         GG.add_edge(n,nbr, weight = minvalue)
# print GG.degree(weight='weight')
# print GG.edges()
# print nx.shortest_path(GG, 1, 3)



# if G.get_edge_data(3,2).has_key("123"):
#     print 123
# G.add_edge(1, 2, weight=1 + G.get_edge_data(1,2)['weight'])
#
# print G.degree(weight="weight")
# print G.degree(2, weight="weight")

import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# z = np.cos(x**2)


# G =nx.random_graphs.barabasi_albert_graph(100,1)   #生成一个BA无标度网络G
# nx.draw(G,node_size = 20)
# plt.show()                            #输出方式2: 在窗口中显示这幅图像

# def aba_sms_test(filename):
#     file = open(filename, 'r')
#
#     num = set()
#     time_stamp = set()
#     line_cnt = 0
#     max = 0
#     min = 10111972131
#     for line in file:
#         line_cnt += 1
#         line_data = line.split(' ')
#         from_id = int(line_data[0])
#         to_id = int(line_data[1])
#         time_stamp = int(line_data[3])
#
#         if max < time_stamp:
#             max = time_stamp
#         if min > time_stamp:
#             min = time_stamp
#
#     # print max(time_stamp)
#     print max
#     print min
#     print line_cnt
#     file.close()
#
#
# # aba_sms_test("E:\\enron\\enron\\out.enron")
# x = time.localtime(989734307)
# print time.strftime('%Y-%m-%d %H:%M:%S',x)
