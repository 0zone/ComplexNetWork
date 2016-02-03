# -*- coding: gb18030 -*-
__author__ = 'yu'

import time
import networkx as nx                   #导入networkx包
import matplotlib.pyplot as plt     #导入绘图包matplotlib（需要安装，方法见第一篇笔记）

GG = nx.Graph()
MG=nx.MultiGraph()
MG.add_weighted_edges_from([(130865333321308000001,2,.5), (1,2,.75), (2,3,.9), (1,4,0.5), (4,3,0.2)])
print MG.degree(weight='weight')
print MG.edges()

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