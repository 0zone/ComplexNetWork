# -*- coding: gb18030 -*-
__author__ = 'yu'
import numpy as np

current_path = "D:\\ComplexNetwork\\"


def get_feature_number(num_file_name, result_file_name, max_line_cnt, smallest_feature_num):
    num_file = open(num_file_name, 'r')
    result_file = open(result_file_name, 'w')

    line_cnt = 0
    for line in num_file:
        data = line.split(" ")
        # if int(data[1]) > smallest_feature_num:
        # if line_cnt > max_line_cnt:
        #     break
        if int(data[1]) > smallest_feature_num:
            result_file.write(data[0] + '\n')
        line_cnt += 1

    result_file.close()
    num_file.close()


def get_number_score(num_file_name, result_file_name, date_index):
    num_file = open(num_file_name, 'r')
    result_file = open(result_file_name, 'w')

    line_cnt = 0
    for line in num_file:
        data = line.split(" ")
        score = data[date_index+1]
        if float(score) < 0.000001:
            # score = "0"
            continue
        result_file.write(data[0] + " " + score + "\n")
        line_cnt += 1

    result_file.close()
    num_file.close()


def get_score_threshold(num_file_name, result_file_name, data_length, thres):
    num_file = open(num_file_name, 'r')
    result_file = open(result_file_name, 'w')
    line_cnt = 0
    for line in num_file:
        line_cnt += 1
        data = line.split(" ")
        if float(data[1]) < (float(data_length)/2):
            continue
        for i in range(5, data_length+2):
            if float(data[i]) > thres:
                result_file.write(data[0] + '\n')
                break

    result_file.close()
    num_file.close()
# get_feature_number(current_path+"result\\gsm_30-0-1805981.txt", current_path+"result\\gsm_30_anomaly.txt", 3000000, 26)
# get_feature_number(current_path+"result\\gsm_30-1805981-2092805.txt", current_path+"result\\gsm_30_anomaly.txt", 3000000, 26)
# get_feature_number(current_path+"result\\gsm_30-2092804-400w.txt", current_path+"result\\gsm_30_anomaly.txt", 3000000, 26)

data_length = 76
thres = 0.6
# get_score_threshold(current_path+"result\\node_score\\gsm_7_anomaly.txt", current_path+"result\\node_score\\gsm_7_" + str(thres) + "anomaly.txt", data_length, thres)
# get_number_score(current_path+"result\\node_score\\gsm_7_anomaly.txt", current_path+"result\\node_score\\gsm_7_" + str(date_index) +"th_anomaly.txt", date_index)
# get_score_threshold(current_path+"result\\node_score\\gsm_7_" + str(date_index) + "th_anomaly.txt")
get_feature_number(current_path+"result\\node_score\\gsm_7_anomaly.txt", current_path+"result\\node_score\\gsm_7_55anomaly.txt",0.1 ,60)