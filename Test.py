# -*- coding: gb18030 -*-
__author__ = 'yu'

import time
import datetime
import networkx as nx                   #导入networkx包
import matplotlib.pyplot as plt     #导入绘图包matplotlib（需要安装，方法见第一篇笔记）
import MySQLdb

current_path = "D:\\ComplexNetwork"
GG = nx.Graph()
# GG.add_weighted_edges_from([ (1,2,4), (1,2,1), (2,3,2),(3,2,1) ,(1,4,2), (4,3,1), (1,3,1), (2,1,1), (5,6,1)])
GG.add_weighted_edges_from([ ('2','2',1),('1','2',4), ('1','4',4),('1','3',1), ('2','3',2),('3','4',1) ,('3','5',2) ,('2','6',2) ,('5','6',2), ('1','1',4), ('1','1',4)])
# GG.add_weighted_edges_from([ (1,2,4)])
# print float(sum(GG.degree(weight='weight').values()))/GG.number_of_nodes()
# print nx.average_clustering(GG)
# print nx.number_connected_components(GG)
# print nx.density(GG)
# print nx.degree_assortativity_coefficient(GG)
# print GG.number_of_edges()
# print 1,2
n = GG.neighbors('1')
n.append('1')
ego = nx.ego_graph(GG, '1' ,radius=1)

print ego.edges()
print ego.number_of_selfloops()
print ego.number_of_nodes()
print ego.number_of_edges()
w = sum(ego.degree(weight='weight').values()) / 2
print w
# print H.edges()
# print H.number_of_edges()
# print H.number_of_nodes()
# # print H.edges(weight='weight')
# print sum(H.degree(weight='weight').values())
# print nx.graph_clique_number(GG)
# print nx.degree_centrality(GG)
# print nx.clustering(GG)
# print nx.betweenness_centrality(GG)
# print nx.pagerank(GG)




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
#  aba_gsm数据迁移到数据库
def aba_gsm_txt2db(file_name):
    aba_begin_date = datetime.datetime(2009, 12, 31)
    file = open(file_name, 'r')

    line_cnt = 0
    num_set = set()
    for line in file:
        line_cnt += 1
        data = line.strip().split(',')

        from_num = data[0]
        to_num = data[9]
        if not to_num.isdigit():
            continue

        year = data[6][0:4]
        month = data[8].split('-')[1].split('月')[0]
        day = data[8].split('-')[0]
        date_time = datetime.datetime(int(year), int(month), int(day))
        date_index = (date_time - aba_begin_date).days
        if date_index > 530:
            continue

        num_set.add(from_num)
        num_set.add(to_num)

        if line_cnt % 10000 == 0:
            print line_cnt

    file.close()
    print len(num_set)

# aba_gsm_txt2db("D:\\数据集\\阿坝\\bf_gsm_call_t_all\\bf_gsm_call_t_all.txt")

# conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
# cur = conn.cursor()
# reslut_file_name = current_path + "\\result\\num.txt"
# file = open(reslut_file_name, 'w')
#
# sql_select = 'SELECT DISTINCT num FROM aba_gsm_30d'
# cur.execute(sql_select)
# result_data = cur.fetchall()
# for row in result_data:
#     file.write(row[0] + '\n')

# import networkx as nx
# G = nx.random_graphs.barabasi_albert_graph(100000,3)   #生成一个n=1000，m=3的BA无标度网络
# degree = nx.degree_histogram(G)          #返回图中所有节点的度分布序列
# x = range(len(degree))                             #生成x轴序列，从1到最大度
# plt.loglog(x,degree,"bo",linewidth=2)           #在双对数坐标轴上绘制度分布曲线
# plt.show()                                                          #显示图表
