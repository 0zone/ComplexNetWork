# -*- coding: gb18030 -*-
__author__ = 'yu'
import numpy as np
import matplotlib.pyplot as plt


def polyfit(x, y, degree):
    results = {}
    coeffs = numpy.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot #准确率
    return results

# degree = 5
# x=[ 1 ,2  ,3 ,4 ,5 ,6]
# y=[ 2.5 ,3.51 ,4.45 ,5.52 ,6.47 ,10]
# z1 = polyfit(x, y, degree)
# print z1
#
# f_x = []
# f_y = []
# for i in range(1, 7):
#     f_x.append(i)
#     pre = 0
#     for p in range(len(z1['polynomial'])):
#         pre += p * i**
#     f_y.append()
#
# plt.plot(x, y, 'ro')
#
# plt.show()

x = np.arange(1, 17, 1)
y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
z1 = np.polyfit(x, y, 5)#用3次多项式拟合
p1 = np.poly1d(z1)
print(p1) #在屏幕上打印拟合多项式
yvals = p1(x)#也可以使用yvals=np.polyval(z1,x)
plot1 = plt.plot(x, y, '*', label='original values')
plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.show()