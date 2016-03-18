__author__ = 'yuenyu'

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

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
            if position%3==0:
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

    plt.scatter(x1,y1, marker = 'x',color = 'red',s=50)
    plt.scatter(x2,y2, marker = 's',c = 'blue',s=50)
    plt.scatter(x3,y3, marker = '+',c = 'green',s=50)
    plt.scatter(x4,y4,marker = 'p',c = 'yellow',s=50)
def var(c,n,p):
    temp = []
    if os.path.exists('E:/data/3network_pro/'+n+'/'+str(c)+'.txt'):
        with open('E:/data/3network_pro/'+n+'/'+str(c)+'.txt', 'r') as f:
            for line in f:
                temp.append(float(line.strip().split("    ")[p]))
        return np.var(temp)/(np.mean(temp)*np.mean(temp))

def pic(x):
    # plt.subplot(221)
    # data('as',x+2,1,7,28,49)
    # plt.legend(('1 days','7 days','28 days','49 days'),loc='uper right',fontsize=15)
    # if x==1:
    #     plt.ylabel('degree',fontsize=25)
    # elif x==2:
    #     plt.ylabel('clustering coeffcient')
    # else :
    #     plt.ylabel('connected component')
    # plt.xlabel('time(days)',fontsize=25)
    # plt.xlim(0,900)
    # plt.xticks(np.arange(0,900,200),fontsize=25)
    # plt.yticks(fontsize=25)

    plt.subplot(221)
    data('sms',x+2,1,24,48,72)
    if x==1:
        plt.ylabel('degree',fontsize=25)
    elif x==2:
        plt.ylabel('clustering coeffcient')
    else :
        plt.ylabel('connected component')
    plt.legend(('1 hours','24 hours','48 hours','72 hours'),loc='uper right',fontsize=15)
    # if x==1:
    #     plt.ylabel('degree')
    # elif x==2:
    #     plt.ylabel('clustering coeffcient')
    # else :
    #     plt.ylabel('connected component')
    plt.xlabel('time(hours)',fontsize=25)
    plt.xlim(0,900)

    plt.xticks(np.arange(0,900,200),fontsize=25)
    plt.yticks(fontsize=25)

    plt.subplot(222)
    data('gsm',x,1,7,28,49)
    plt.legend(('1 days','7 days','28 days','49 days'),loc='uper right',fontsize=15)
    # if x==1:
    #     plt.ylabel('degree')
    # elif x==2:
    #     plt.ylabel('clustering coeffcient')
    # else :
    #     plt.ylabel('connected component')
    plt.xlabel('time(days)',fontsize=25)
    plt.xlim(0,900)

    plt.xticks(np.arange(0,900,200),fontsize=25)
    plt.yticks(fontsize=25)

    # plt.subplot(223)
    # x1=range(1,100)
    # y1=[]
    # for i in x1:
    #     y1.append(var(i,'as',x))
    # plt.scatter(x1,y1,color = 'red',s=50)
    # plt.ylabel('$\sigma$',fontsize=25)
    # plt.xlabel('time scale',fontsize=25)
    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=25)

    plt.subplot(223)
    x1=range(1,60)
    y1=[]
    for i in x1:
        y1.append(var(i,'sms',x))
    plt.scatter(x1,y1,color = 'red',s=50)
    plt.ylabel('$\sigma$',fontsize=25)
    plt.xlabel('time scale',fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.subplot(224)
    x1=range(1,100)
    y1=[]
    for i in x1:
        y1.append(var(i,'gsm',x))
    plt.scatter(x1,y1,color = 'red',s=50)
    plt.xlabel('time scale',fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.savefig('pic'+str(x)+'', dpi = 500)
    plt.show()
pic(3)






