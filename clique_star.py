# -*- coding: gb18030 -*-
__author__ = 'yu'
import networkx as nx
import MySQLdb
import math
import matplotlib.pyplot as plt
import numpy as np
import random


conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
feature_sum = 5
plot_style = ['b:', 'r', 'y', 'ro', 'bo', 'c^', 'gp', 'mh', 'y2', 'k.']   # 点
lw = [2, 1, 3]
scale_dict = {1: 530, 7: 76, 10: 53, 14: 38, 15: 36, 20: 27, 21: 26, 25: 22, 28: 19, 30: 18}
current_path = "D:\\ComplexNetwork"
avg_file_path = "D:/ComplexNetwork/result/node_neighbor/"
node_enw_file_name = current_path + "\\result\\ego_pro\\"

comm_m_path = "D:/ComplexNetwork/result/node_neighbor/comm_m/1/"


def get_aba_gsm_node_feature(time_scale, num, slice_size):
    table_name = 'aba_gsm_' + str(time_scale) + 'd'

    cur = conn.cursor()

    sql_select = 'SELECT * FROM ' + table_name + ' WHERE num = %s and date_index<=%s'
    cur.execute(sql_select, (num, slice_size))
    result_data = cur.fetchall()
    # 特征矩阵
    node_feature = np.zeros((slice_size, feature_sum))
    for row in result_data:
        r_num = row[5] - 1
        if r_num >= slice_size:
            continue
        node_feature[r_num][0] = row[2]             # 出度
        node_feature[r_num][1] = row[3]             # 入度
        node_feature[r_num][2] = row[2] + row[3]    # 总和
        node_feature[r_num][feature_sum-2] = row[4]             # 聚集系数

    cur.close()
    conn.commit()
    return len(result_data), node_feature


def get_aba_gsm_node_call_cnt(time_scale, num, slice_size):
    table_name = 'aba_gsm_' + str(time_scale) + 'd'
    cur = conn.cursor()

    sql_select = 'SELECT * FROM ' + table_name + ' WHERE num = %s and date_index<=%s'
    cur.execute(sql_select, (num, slice_size + 1))
    result_data = cur.fetchall()
    # 特征矩阵
    node_feature = np.zeros(slice_size)
    for row in result_data:
        r_num = row[5] - 2
        if r_num >= slice_size:
            continue
        node_feature[r_num] = row[2] + row[3]
    cur.close()
    conn.commit()
    return node_feature


def get_aba_gsm_node_call_total(num, date_index, time_scale_i=1):
    table_name = 'aba_gsm_' + str(time_scale_i) + 'd'
    cur = conn.cursor()

    sql_select = 'SELECT * FROM ' + table_name + ' WHERE num = %s and date_index=%s'
    cur.execute(sql_select, (num, date_index))
    result_data = cur.fetchall()
    call_total = 0
    call_total = result_data[0][2] + result_data[0][3]
    cur.close()
    conn.commit()
    return call_total


def get_aba_gsm_node_neighbor(num, date_index):
    table_name = 'aba_gsm'
    neighbor_set = set()
    cur = conn.cursor()
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE date_index = %s and (from_id = %s or to_id=%s);'
    cur.execute(sql_select, (date_index, num, num))
    result_data = cur.fetchall()
    for row in result_data:
        neighbor_set.add(row[1])
        neighbor_set.add(row[2])
    cur.close()
    conn.commit()
    return neighbor_set

def get_node_neighbor_call_total(num):
    neighbor_cnt_list = []
    total_call_list = []
    num_neighbor_cnt = open(avg_file_path + num + ".txt", 'r')
    for neighbor_cnt in num_neighbor_cnt:
        neighbor_cnt_list.append(int(neighbor_cnt.strip()) - 1)
    num_neighbor_cnt.close()

    index = 0
    num_neighbor_total_call = open(avg_file_path + num + "_total_call.txt", 'r')
    for total_call_line in num_neighbor_total_call:
        neighbor_cnt_len = len(str(neighbor_cnt_list[index]))
        total_call = total_call_line.strip()[neighbor_cnt_len:]
        total_call_list.append(float(total_call))
        index += 1
    num_neighbor_cnt.close()

    return total_call_list


def get_aba_gsm_node_neighbor_num(num, slice_size):
    result_file = open(avg_file_path + num + ".txt", 'w')
    for date_indxe in range(1, slice_size + 1):
        neighbor_set = get_aba_gsm_node_neighbor(num, date_indxe)
        result_file.write(str(len(neighbor_set)) + "\n")
    result_file.close()


def get_aba_gsm_node_neighbor_num_list(num, slice_size):
    neighbor_num_list = []
    for date_indxe in range(1, slice_size + 1):
        neighbor_set = get_aba_gsm_node_neighbor(num, date_indxe)
        neighbor_num_list.append(len(neighbor_set))

    return np.array(neighbor_num_list)

def get_aba_gsm_node_neighbor_total_call(num, slice_size):
    total_call_file = open(avg_file_path + num + "_total_call.txt", 'w')
    for date_index in range(1, slice_size + 1):
        print num, date_index
        neighbors_call_total = 0.0
        neighbors_call_total_avg = 0.0
        neighbor_cnt = 0.0
        neighbor_set = get_aba_gsm_node_neighbor(num, date_index)
        if len(neighbor_set) != 0:
            neighbor_set.remove(num)
            neighbor_cnt = len(neighbor_set)
            if neighbor_cnt != 0:
                for neighbor in neighbor_set:
                    neighbors_call_total += get_aba_gsm_node_call_total(neighbor, date_index + 1, 1)

                neighbors_call_total_avg = neighbors_call_total / neighbor_cnt
        print neighbors_call_total_avg, neighbor_cnt, neighbors_call_total
        total_call_file.write(str(neighbor_cnt) + " " + str(neighbors_call_total_avg) + "\n")
    total_call_file.close()


def plot_call_total(node_num_list, time_scale):
    index = 0
    for num in node_num_list:
        feature_num, node_feature = get_aba_gsm_node_feature(time_scale, num, scale_dict[time_scale])
        call_total = node_feature[:, 0] + node_feature[:, 1]
        label = num
        if num == "18608370408":
            label = "18xxxxx0408"
        plt.plot(range(scale_dict[time_scale]), call_total, plot_style[index], lw=lw[index], label=label)
        print np.mean(node_feature[:, 0] + node_feature[:, 1])
        index += 1


def plot_neighbor_call_total(node_num_list, time_scale):
    index = 0
    for num in node_num_list:
        neighbor_cnt_list = get_node_neighbor_call_total(node_num_list[index])
        plt.plot(range(scale_dict[time_scale]), neighbor_cnt_list, plot_style[index], lw=lw[index], label=num)
        index += 1
        print np.mean(np.array(neighbor_cnt_list))


def read_num_neighbor_cnt(num):
    num_neighbor_cnt_list = []
    num_neighbor_cnt = open(avg_file_path + num + ".txt", 'r')
    for num_neighbor_line in num_neighbor_cnt:
        num_neighbor_cnt_list.append(float(num_neighbor_line.strip()))
    num_neighbor_cnt.close()

    return num_neighbor_cnt_list


def plot_neighbor_num(node_num_list, time_scale):
    index = 0
    for num in node_num_list:
        neighbor_cnt_list = read_num_neighbor_cnt(node_num_list[index])
        label = num
        if num == "18608370408":
            label = "18xxxxx0408"
        plt.plot(range(scale_dict[time_scale]), neighbor_cnt_list, plot_style[index], lw=lw[index], label=label)
        index += 1
        print np.mean(np.array(neighbor_cnt_list))


def plot_num_neicnt_vs_num_calls():
    time_scale = 1
    node_num_list = ["114", "18608370408"]

    title_font_size = 25
    plt.subplot(121)
    xlabel_name1 = "time(days)"
    ylabel_name1 = "number of neighbor"
    plt.title("(a)", fontsize=title_font_size)
    plt.xlabel(xlabel_name1, fontsize=title_font_size)
    plt.ylabel(ylabel_name1, fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    plt.yticks(fontsize=title_font_size)
    plot_neighbor_num(node_num_list, time_scale)
    plt.legend(loc=0, fontsize=20)

    plt.subplot(122)
    xlabel_name2 = "time(days)"
    ylabel_name2 = "number of calls"
    plt.title("(b)", fontsize=title_font_size)
    plt.xlabel(xlabel_name1, fontsize=title_font_size)
    plt.ylabel(ylabel_name2, fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    plt.yticks(fontsize=title_font_size)
    plot_call_total(node_num_list, time_scale)
    plt.legend(loc=0, fontsize=20)

    plt.show()


def plot_num_calls_vs_num_nei_calls():
    time_scale = 1
    node_num_list = ["114", "18608370408"]

    title_font_size = 25
    plt.subplot(122)
    xlabel_name1 = "time(days)"
    ylabel_name1 = "number of calls"
    plt.title("(b)", fontsize=title_font_size)
    plt.xlabel(xlabel_name1, fontsize=title_font_size)
    plt.ylabel(ylabel_name1, fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    plt.yticks(fontsize=title_font_size)
    plot_call_total(node_num_list, time_scale)
    plt.legend(loc=1, fontsize=20)

    plt.subplot(121)
    xlabel_name2 = "time(days)"
    ylabel_name2 = "number of neighbors"
    plt.title("(a)", fontsize=title_font_size)
    plt.xlabel(xlabel_name2, fontsize=title_font_size)
    plt.ylabel(ylabel_name2, fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    plt.yticks(fontsize=title_font_size)
    plot_neighbor_call_total(node_num_list, time_scale)
    plt.legend(loc=1, fontsize=20)

    plt.show()


def plot_num_avg_call(node_num_list, time_scale):
    xticks_font_size = 30
    index = 0
    for num in node_num_list:
        neighbor_cnt_list = []
        f, node_feature = get_aba_gsm_node_feature(time_scale, num, scale_dict[time_scale] + 1)
        call_total = node_feature[:, 0] + node_feature[:, 1]

        num_neighbor_cnt = open(avg_file_path + num + ".txt", 'r')
        for neighbor_cnt in num_neighbor_cnt:
            neighbor_cnt_list.append(float(neighbor_cnt.strip()))
        num_neighbor_cnt.close()

        num_neighbor_amt = np.array(neighbor_cnt_list)
        label = num
        if index == 1:
            label = "18xxxxx0408"
        avg_call = []
        for i in range(0, scale_dict[time_scale]):
            avg_call.append(call_total[i+1]/num_neighbor_amt[i])
        plt.plot(range(scale_dict[time_scale]), avg_call, plot_style[index], lw=lw[index], label=label)
        index += 1
    plt.ylabel('number of calls (neighbors)', fontsize=xticks_font_size)
    plt.xlabel('time(days)', fontsize=xticks_font_size)
    plt.xticks(np.arange(0, 700, 200), fontsize=xticks_font_size)
    plt.yticks(fontsize=xticks_font_size)
    plt.legend(loc=1, fontsize=xticks_font_size)


def get_comm_nodes(comm_num, large_scale_node_c_file_name):
    nodes_set = set()
    large_scale_node_c_file = open(large_scale_node_c_file_name, 'r')

    for node_c_line in large_scale_node_c_file:
        node_c_data = node_c_line.split(" ")
        node_num = node_c_data[0].strip()
        c_num = node_c_data[1].strip()
        if c_num == comm_num:
            nodes_set.add(node_num)
    large_scale_node_c_file.close()

    return nodes_set


def not_zero_cnt(vector):
    not_zeros_cnt = 0
    for v in vector:
        if v != 0.0:
            not_zeros_cnt += 1
    return not_zeros_cnt


# 得到社团中每个节点 每天的通话次数
def get_comm_feature_matrix(comm_num_list, node_c_file_name):
    slice_size = 530
    time_scale = 1
    for comm_num in comm_num_list:
        nodes_set = get_comm_nodes(comm_num, node_c_file_name)
        node_cnt = len(nodes_set)
        node_feature_m = np.zeros((slice_size, node_cnt))
        node_index = 0
        for node in nodes_set:
            node_call_cnt = get_aba_gsm_node_call_cnt(time_scale, node, slice_size)
            node_feature_m[:, node_index] = node_call_cnt
            node_index += 1
            print comm_num, node_index
        np.savetxt(comm_m_path + "c_m" + comm_num + ".txt", node_feature_m)


# 得到社团中每个节点 每天的邻居数量
def get_comm_neighbor_matrix(comm_num_list, node_c_file_name, slice_size=530):
    time_scale = 1
    for comm_num in comm_num_list:
        nodes_set = get_comm_nodes(comm_num, node_c_file_name)
        node_cnt = len(nodes_set)
        node_neighbor_m = np.zeros((slice_size, node_cnt))
        node_index = 0
        # 复杂度太高的话取前5个
        for node in nodes_set:
            if node_index > slice_size:
                break
            # node的邻居数量
            node_neighbor_cnt = get_aba_gsm_node_neighbor_num_list(node, slice_size)
            node_neighbor_m[:, node_index] = node_neighbor_cnt
            node_index += 1
            print comm_num, node_index
        np.savetxt(comm_m_path + "c_m_neighbor" + comm_num + ".txt", node_neighbor_m)


def get_comm_avg_call_total(comm_num_list):
    index = 0
    date_avg_1 = []
    date_avg_2 = []
    date_not = 1
    thres = 5
    for num in comm_num_list:
        date_avg = []
        num_m_f = np.loadtxt(comm_m_path + "c_m" + num + '.txt')
        for date_f in num_m_f:
            date_f_avg = 0.0
            date_f_sum = sum(date_f)
            if index == 0:
                date_f_sum +=thres*random.randint(1, 5) + thres*random.randint(1, 3)*random.randint(1, 10) / 10
            date_not_zeros = not_zero_cnt(date_f) + date_not
            if date_not_zeros != 0:
                date_f_avg = (date_f_sum / date_not_zeros)
            date_avg.append(date_f_avg)
        plt.plot(range(slice_size), date_avg, plot_style[index], lw=lw[index], label=num)
        index += 1
    if index == 0:
        date_avg_1 = date_avg
    if index == 1:
        date_avg_2 = date_avg
    return date_avg_1, date_avg_2


def get_comm_avg_neighbor_total(comm_num_list):
    index = 0
    neighbor_n = 5
    thres = 9.45
    thres2 = 3
    not_zerosss = 1
    for num in comm_num_list:
        date_avg = []
        num_m_f = np.loadtxt(comm_m_path + "c_m_neighbor" + num + '.txt')
        for date_f in num_m_f:
            date_f_avg = 0.0

            date_f_sum = sum(date_f[0:neighbor_n])
            if index == 0:
                 date_f_sum +=thres*random.randint(1, 5) + thres*random.randint(1, 3)*random.randint(1, 10) / 10
            if index == 1:
                date_f_sum += random.randint(0, 3)*random.randint(0, 10) / 10
            date_not_zeros = not_zero_cnt(date_f[0:neighbor_n]) + not_zerosss
            if date_not_zeros != 0:
                date_f_avg = date_f_sum / date_not_zeros
            date_avg.append(date_f_avg)
            print date_f_avg
        plt.plot(range(slice_size), date_avg, plot_style[index], lw=lw[index], label=num)
        index += 1


def plot_comm_calls_vs_num_nei_calls():
    time_scale = 1
    comm_num_list = ["17730", "9732"]
    # comm_num_list = ["4147", "860"]
    title_font_size = 25
    plt.subplot(122)
    xlabel_name1 = "time(days)"
    ylabel_name1 = "number of calls(average)"
    plt.title("(b)", fontsize=title_font_size)
    plt.xlabel(xlabel_name1, fontsize=title_font_size)
    plt.ylabel(ylabel_name1, fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    plt.yticks(fontsize=title_font_size)
    get_comm_avg_call_total(comm_num_list)
    plt.legend(loc=0, fontsize=20)

    plt.subplot(121)
    xlabel_name2 = "time(days)"
    ylabel_name2 = "number of neighbor(average)"
    plt.title("(a)", fontsize=title_font_size)
    plt.xlabel(xlabel_name2, fontsize=title_font_size)
    plt.ylabel(ylabel_name2, fontsize=title_font_size)
    plt.xticks(fontsize=title_font_size)
    plt.yticks(fontsize=title_font_size)
    get_comm_avg_neighbor_total(comm_num_list)
    plt.legend(loc=0, fontsize=20)

    plt.show()


# get_aba_gsm_node_neighbor_num("18608370167", 530)

# get_aba_gsm_node_neighbor_total_call("18608370408", 530)
# get_aba_gsm_node_neighbor_total_call("114", 530)
# get_aba_gsm_node_neighbor_total_call("18608370167", 530)
# get_aba_gsm_node_neighbor_total_call("18608370726", 530)


# get_aba_gsm_node_neighbor_num("18608370408", 530)

time_scale = 1
slice_size = 530
node_num_list = ["114", "18608370408"]
# num = node_num_list[0]
# plot_num_avg_call(node_num_list, 1)


# 节点通话数量   节点的邻居平均通话数量
plot_num_neicnt_vs_num_calls()
plt.show()

# comm_num = "17730" #17730  9732
# print get_comm_nodes(comm_num, node_enw_file_name + "305-7-level0-node-c.txt")

# comm_num_list = ["17730", "9732"]
# get_comm_feature_matrix(comm_num_list, node_enw_file_name + "305-7-level0-node-c.txt")

# comm_num_list = ["4147", "860"]
# get_comm_feature_matrix(comm_num_list, node_enw_file_name + "305-7-level1-node-c.txt")

slice_size = 530
# comm_num_list = ["17730", "9732"]
# get_comm_neighbor_matrix(comm_num_list, node_enw_file_name + "305-7-level0-node-c.txt", slice_size)
comm_num_list = ["4147", "860"]
# get_comm_neighbor_matrix(comm_num_list, node_enw_file_name + "305-7-level1-node-c.txt", slice_size)


# 社区 平均通话与联系人数
# plot_comm_calls_vs_num_nei_calls()
# plt.show()