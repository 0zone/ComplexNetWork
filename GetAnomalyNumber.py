# -*- coding: gb18030 -*-
__author__ = 'yu'


current_path = "D:\\ComplexNetwork\\"


def get_feature_number(num_file_name, result_file_name, smallest_feature_num):
    num_file = open(num_file_name, 'r')
    result_file = open(result_file_name, 'a')

    line_cnt = 0
    for line in num_file:
        data = line.split(" ")
        if int(data[1]) > smallest_feature_num:
            result_file.write(line)
        line_cnt += 1

    result_file.close()
    num_file.close()

get_feature_number(current_path+"result\\gsm_10-1-400w.txt", current_path+"result\\gsm_10_feature_num.txt", 26)
