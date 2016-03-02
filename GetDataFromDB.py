# -*- coding:gb2312 -*-
__author__ = 'yu'

import MySQLdb


def get_aba_sms_data_from_db(time_scale):
    aba_sms_path = "D:\\数据集\\阿坝\\sms\\"
    aba_sms_max_num = 9330493
    table_name = 'aba_sms'
    min_time_stamp = 1291737600
    max_time_stamp = 1294415993

    # 获取最小时间戳的时间，精确到小时
    min_time_stamp -= (min_time_stamp % 3600)

    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    cur = conn.cursor()
    sql_select = 'SELECT * FROM ' + table_name + ' WHERE time_stamp >= %s and time_stamp < %s'

    index = 0
    for begin_timestamp in range(min_time_stamp, max_time_stamp, time_scale):
        index += 1
        end_timestamp = begin_timestamp + time_scale
        cur.execute(sql_select, (begin_timestamp, end_timestamp))

        result_data = cur.fetchall()
        # 构建网络
        # construct
        file_name = aba_sms_path + str(time_scale) + "\\" + str(index) + ".txt"
        file = open(file_name, 'w')
        file.write(str(aba_sms_max_num) + '    ' + str(len(result_data)) + '\n')
        for row in result_data:
            file.write(str(row[1]) + '    ' + str(row[2]) + '\n')
        file.close()
    cur.close()
    conn.commit()
    conn.close()

    return result_data


get_aba_sms_data_from_db(3600)