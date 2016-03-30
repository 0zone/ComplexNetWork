__author__ = 'yuenyu'

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def data(n,s,m1,m2,m3,m4):
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    y1=[]
    y2=[]
    y3=[]
    y4=[]
    with open('E:/data/3network_pro/'+n+'/'+str(m1)+'.txt', 'r') as f:
        for position,line in enumerate(f):
            x1.append(float(line.strip().split("    ")[0])*m1)
            y1.append(float(line.strip().split("    ")[s]))
    with open('E:/data/3network_pro/'+n+'/'+str(m2)+'.txt', 'r') as f:
        for line in f:
            x2.append(float(line.strip().split("    ")[0])*m2)
            y2.append(float(line.strip().split("    ")[s]))
    with open('E:/data/3network_pro/'+n+'/'+str(m3)+'.txt', 'r') as f:
        for line in f:
            x3.append(float(line.strip().split("    ")[0])*m3)
            y3.append(float(line.strip().split("    ")[s]))
    with open('E:/data/3network_pro/'+n+'/'+str(m4)+'.txt', 'r') as f:
        for line in f:
            x4.append(float(line.strip().split("    ")[0])*m4)
            y4.append(float(line.strip().split("    ")[s]))
    plt.scatter(x1,y1, marker = 'x',color = 'red')
    plt.scatter(x2,y2, marker = 's',c = 'blue')
    plt.scatter(x3,y3, marker = '+',c = 'green')
    plt.scatter(x4,y4,marker = 'p',c = 'yellow')

def pic(x):
    plt.subplot(131)
    data('as',x,1,7,28,49)
    plt.legend(('1 days','7 days','28 days','49 days'),loc='uper right',fontsize=5)
    plt.ylabel('degree')
    plt.xlabel('time(days)')

    plt.subplot(132)
    data('sms',x,1,24,48,72)
    plt.legend(('1 hours','24 hours','48 hours','72 hours'),loc='uper right',fontsize=5)
    plt.ylabel('degree')
    plt.xlabel('time(hours)')

    plt.subplot(133)
    data('gsm',x,1,7,28,49)
    plt.legend(('1 days','7 days','28 days','49 days'),loc='uper right',fontsize=5)
    if x==1:
        plt.ylabel('degree')
    elif x==2:
        plt.ylabel('clustering coeffcient')
    else :
        plt.ylabel('connected component')
    plt.xlabel('time(days)')
    plt.savefig('pic'+str(x)+'', dpi = 500)
    plt.show()
pic(1)
# def var(c,n,p):
#     temp = []
#     with open('E:/data/3network_pro/'+n+''+str(c)+'.txt', 'r') as f:
#         for line in f:
#             temp.append(float(line.strip().split("    ")[p]))
#     return np.var(temp)/(np.mean(temp)*np.mean(temp))
#
# x1=range(1,31)
# y1=[]
# for i in x1:
#     y1.append(var(i,'as',1))
# plt.scatter(x1,y1,color = 'red')
# plt.title('Degree-time scale ')
# plt.ylabel('Degree')
# plt.xlabel('time scale')
# plt.savefig('Degree_time.png', dpi = 500)
# plt.show()


