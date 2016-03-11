# -*- coding: gb18030 -*-
__author__ = 'yu'
from community import *
import networkx as nx
import matplotlib.pyplot as plt
import MySQLdb


def get_gsm_network_node_property(begin_date, time_scale):
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

G = get_gsm_network_node_property(1, 1)
print 1
dendo = generate_dendogram(G)
for level in range(len(dendo) - 1):
    print "partition at level", level, "is", #partition_at_level(dendo, level)
# G=nx.erdos_renyi_graph(20, 0.05)
# G = nx.Graph()
# G.add_weighted_edges_from([ (1,2,4), (1,2,1), (2,3,2),(3,2,1) ,(1,4,2), (4,3,1), (1,3,1), (2,1,1), (5,6,1)])
partition = best_partition(G)
# print partition
# nx.draw(G, nx.spring_layout(G), alpha=0.5, node_size=50)
# plt.show()
# plt.close()
# dendo = generate_dendogram(G)
# for level in range(len(dendo) - 1):
#     print "partition at level", level, "is", partition_at_level(dendo, level)
#     ind = induced_graph(partition_at_level(dendo, level), G)
#     nx.draw(ind, nx.spring_layout(ind), alpha=0.5, node_size=50)
#     plt.show()
#first compute the best partition
# n = 5
# g = nx.complete_graph(2*n)
# part = dict([])
# for node in g.nodes():
#     part[node] = node % 2
# print part

# goal = nx.Graph()
# goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])
# print nx.is_isomorphic(ind, goal)
#
# #drawing
# size = float(len(set(partition.values())))
# pos = nx.spring_layout(G)
# count = 0.
# for com in set(partition.values()) :
#     count = count + 1.
#     list_nodes = [nodes for nodes in partition.keys()
#                                 if partition[nodes] == com]
#     nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
#                                 node_color = str(count / size))
#
#
# nx.draw_networkx_edges(G,pos, alpha=0.5)
# plt.show()