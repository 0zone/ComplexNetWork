__author__ = 'yuenyu'

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
def drop_zeros(a_list):
    return [i for i in a_list if i > 0]

def log_binning(x, bin_count=10):
    min_x = math.log10(min(drop_zeros(x)))
    max_x = math.log10(max(drop_zeros(x)))
    bins = np.logspace(min_x, max_x, num=bin_count)

    return bins


def bining_data(x_vec, y_vec, bin_count=11):
    bins = log_binning(x_vec, bin_count)
    bin_x = [[] for i in range(bin_count-1)]
    bin_y = [[] for i in range(bin_count-1)]

    for x_index in range(0, len(x_vec)):
        x = x_vec[x_index]
        bin_index = 0
        for bin_index in range(0, bin_count):
            if bins[bin_index] > x:
                break
        bin_x[bin_index-1].append(x_vec[x_index])
        bin_y[bin_index-1].append(y_vec[x_index])
    return bin_x, bin_y

def pic(c,n):
    G = nx.Graph()
    if c=='gsm':
        with open('E:/data/degree-distribution/'+c+'/305-7-level'+str(n)+'.txt', 'r') as f:
            for position, line in enumerate(f):
                if line.strip().split(' ')[0]!=line.strip().split(' ')[1]:
                    u= line.strip().split(' ')[0]
                    n=line.strip().split(' ')[1]
                    G.add_edge(u, n)
    else:
        with open('E:/data/degree-distribution/'+c+'/49-24-level'+str(n)+'.txt', 'r') as f:
            for position, line in enumerate(f):
                if line.strip().split(' ')[0]!=line.strip().split(' ')[1]:
                    u= line.strip().split(' ')[0]
                    n=line.strip().split(' ')[1]
                    G.add_edge(u, n)
    degree_hist = nx.degree_histogram(G)
    x = range(len(degree_hist))[1:]
    print(degree_hist)

    y = [float(i+1) / float(sum(degree_hist)) for i in degree_hist[1:]]
    plt.loglog(x, y, 'bo', linewidth = 2)
    plt.title('Degree Distribution')
    plt.ylabel('Probability')
    plt.xlabel('Degree')
    return x, y


plt.subplot(221)
pic('gsm','')
plt.show()

d3,p3=pic('gsm',1)
ls_fit = np.polyfit(np.log10(d3), np.log10(p3), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'g', linewidth = 2)
d4,p4=pic('gsm',2)
ls_fit = np.polyfit(np.log10(d4), np.log10(p4), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'k', linewidth = 2)
d5,p5=pic('gsm',3)
ls_fit = np.polyfit(np.log10(d5), np.log10(p5), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'y', linewidth = 2)

plt.subplot(122)
d1,p1=pic('sms','')
ls_fit = np.polyfit(np.log10(d1), np.log10(p1), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'b', linewidth = 2)
d2,p2=pic('sms',0)
ls_fit = np.polyfit(np.log10(d2), np.log10(p2), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'r', linewidth = 2)
d3,p3=pic('sms',1)
ls_fit = np.polyfit(np.log10(d3), np.log10(p3), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'g', linewidth = 2)
d4,p4=pic('sms',2)
ls_fit = np.polyfit(np.log10(d4), np.log10(p4), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'k', linewidth = 2)
d5,p5=pic('sms',3)
ls_fit = np.polyfit(np.log10(d5), np.log10(p5), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'y', linewidth = 2)

d6,p6=pic('sms',4)
ls_fit = np.polyfit(np.log10(d6), np.log10(p6), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'y', linewidth = 2)

d7,p7=pic('sms',5)
ls_fit = np.polyfit(np.log10(d7), np.log10(p7), 1)
print ls_fit
ls_x = range(1, 10000)
ls_y = []
for x in ls_x:
    ls_y.append((10**ls_fit[1])*(x**ls_fit[0]))
plt.loglog(ls_x, ls_y, 'y', linewidth = 2)
plt.title('Degree Distribution')
plt.ylabel('Probability')
plt.xlabel('Degree')
plt.show()