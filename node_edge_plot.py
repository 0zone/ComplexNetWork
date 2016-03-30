# -*- coding: gb18030 -*-
__author__ = 'yu'

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
plot_style = ['g', 'b', 'r', 'ro', 'bs', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点


node_cnt_list = []
file = open('E:/data/3network_pro/sms'+ '/' + str(1)+'.txt', 'r')
for line in file:

    node_cnt = float(line.split("    ")[2])
    node_cnt_list.append(node_cnt)

print node_cnt_list
plt.plot(range(0, len(node_cnt_list)), node_cnt_list, plot_style[1])
plt.show()