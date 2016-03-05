# -*- coding:gb2312 -*-
__author__ = 'yu'

# 迁移AS-733数据

import time
import os
import MySQLdb
import datetime

as_delimiter = '\t'
begin_line_num = 5
as_data_path = "/Users/jinyu/paper/data/as-733/"
as_table_name = 'as733'

aba_delimiter = ','
aba_begin_date = datetime.datetime(2009, 12, 31)
aba_max_date_index = 530
aba_data_path = "/Users/jinyu/paper/data/aba/bf_gsm_call_t_all/bf_gsm_call_t_all.txt"

aba_gsm_data_path = "D:\\数据集\\阿坝\\bf_gsm_call_t_all\\bf_gsm_call_t_all.txt"
aba_sms_data_path = "D:\\数据集\\阿坝\\sms.txt"


# 迁移AS文件夹中的所有数据
def as_migrate_data(file_path):
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    date_index = 0
    path_dir = os.listdir(file_path)
    for filename in path_dir:
        date_index += 1
        print filename
        as_txt2db(conn, file_path + filename, date_index)
    conn.close()


# AS txt数据迁移到数据库
def as_txt2db(conn, filename, date):
    file = open(filename, 'r')
    line_cnt = 0
    date_time = filename[-12:-4]

    rows = []
    for line in file:
        line_cnt += 1
        if line_cnt < begin_line_num:
            continue
        data = line.strip().split(as_delimiter)
        from_id = int(data[0])
        to_id = int(data[1])
        rows.append((from_id, to_id, date, date_time))

    # 存入数据库
    cur = conn.cursor()
    sql_insert = 'INSERT INTO ' + as_table_name + ' (from_id, to_id, date_index, date_time) VALUES(%s, %s, %s,%s);'
    cur.executemany(sql_insert, rows)

    conn.commit()
    cur.close()


# aba_gsm数据迁移到数据库
def aba_gsm_txt2db(file_name):
    aba_gsm_table_name = 'aba_gsm'
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    file = open(file_name, 'r')
    sql_insert = 'INSERT INTO ' + aba_gsm_table_name + ' (from_id, to_id, date_index, date_time) VALUES(%s, %s, %s,%s);'

    rows = []
    line_cnt = 0
    for line in file:
        line_cnt += 1
        data = line.strip().split(aba_delimiter)

        year = data[6][0:4]
        month = data[8].split('-')[1].split('月')[0]
        day = data[8].split('-')[0]
        date_time = datetime.datetime(int(year), int(month), int(day))
        date_index = (date_time - aba_begin_date).days

        from_num = data[0]
        to_num = data[9]
        if not to_num.isdigit():
            continue
        if date_index > aba_max_date_index:
            continue

        rows.append((from_num, to_num, date_index, date_time.strftime("%Y%m%d")))
        if line_cnt % 10000 == 0:
            # 存入数据库
            cur = conn.cursor()
            cur.executemany(sql_insert, rows)
            conn.commit()
            cur.close()
            rows = []               # 清空
            print line_cnt
    # 存入数据库
    cur = conn.cursor()
    cur.executemany(sql_insert, rows)
    conn.commit()
    cur.close()

    file.close()
    conn.close()


def aba_sms_txt2db(filename):
    file = open(filename, 'r')
    conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="network", charset="utf8")
    line_cnt = 1
    rows = []
    file.readline()
    for line in file:
        line_cnt += 1
        line_data = line.split('\t')
        date_time = line_data[0]

        from_id = int(line_data[1].strip())
        to_id = int(line_data[2].strip())
        date_time = time.strptime(date_time, "%Y-%m-%d %H:%M:%S")
        date_index = int(time.mktime(date_time))
        if date_index < 1291737600:
            continue
        date_index = (date_index-1291737600) / 3600 + 1

        rows.append((from_id, to_id, date_index))
        if line_cnt % 100000 == 0:
            cur = conn.cursor()
            sql_insert = 'INSERT INTO aba_sms' + ' (from_id, to_id, date_index) VALUES(%s, %s,%s);'
            cur.executemany(sql_insert, rows)
            conn.commit()
            cur.close()
            rows = []

    cur = conn.cursor()
    sql_insert = 'INSERT INTO aba_sms' + ' (from_id, to_id, date_index) VALUES(%s, %s,%s);'
    cur.executemany(sql_insert, rows)
    conn.commit()
    cur.close()

    conn.close()
    file.close()

aba_gsm_txt2db(aba_gsm_data_path)