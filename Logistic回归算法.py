"""Logistic回归算法"""


"""
Logistic 回归的一般过程
1、收集数据
3、分析数据
4、训练数据：时间长，目的是为了找到最佳的分类回归系数
5、测试数据
6、使用算法

梯度上升法：
（1）每个回归系数初始化为1
（2）重复R次
            计算整个数据集的梯度
            使用alpha * gradient 更新回归系数的变量
            返回回归系数
优点：计算代价不高，易于理解和实现
缺点：容易欠拟合，分类精度可能不高
"""
#Logistic回归梯度上升优化算法
# import numpy as np
# import matplotlib.pyplot as plt

#求f（x）= -X^2 + 4X  最大值对应的X值
"""
1、先求导    f'(x) = -2X + 4
2、最优化算法    Xi+1 = Xi + a [af（Xi）/Xi]       a为步长，控制更新幅度
"""
"""
自己写1
def hanshu():
    def hanshu1(x):
        return -2*x + 4
    x = -1
    x1 = 0
    alpha = 0.02
    presision= 0.0000000001

    while abs(x1 - x)>presision:
        x = x1
        x1 = x+alpha*hanshu1(x)
    print(x1)

if __name__ == '__main__':
    hanshu()
    
    
自己练习求:
f(X)= -5X^2 -10X

def test():
    def test1(X):
        return -10*X -10
    X = -1
    X1 = 0
    alpha = 0.01
    presision = 0.000000001

    while abs(X1-X)>presision:
        X = X1
        X1 = X+alpha*test1(X)
    print(X1)

if __name__ == '__main__':
    test()

 
别人代码
def Gradient_Ascent_test():
    def f_prime(x_old):#f(x)的导数
        return -2*x_old +4
    x_old = -1#初始值，给定一个小于x_new的值
    x_new = 0#梯度上升算法初始值，即从（0，0）开始
    alpha = 0.01#步长，也就是学习速率，控制更新的幅度
    presision = 0.00000001#精度，也就是更新阙值

    while abs(x_new-x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)
if __name__ == '__main__':
    Gradient_Ascent_test()
"""


"""
一定要看数学推导的公式：梯度上升法，要搞明白原理否则只是徒劳
见笔记
"""
"""
一、数据准备
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import *
def loadDataSet():
    dataMat = []#创建数据列表
    labelMat = []#创建标签列表
    fr = open('testSet.txt')#打开文件
    for line in fr.readlines():#逐行读取文件
        lineArr = line.strip().split()#去除回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])#添加数据
        labelMat.append(int(lineArr[2]))#添加标签
    fr.close()#关闭文件
    return dataMat, labelMat


"""
二、训练算法

求解回归系数W1,W2,W3
"""

from numpy import *


def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)#转换成numpy样式的mat
    labelMat = np.mat(classLabels).transpose()#转换成numpy样式的mat， 并进行转置
    m, n = np.shape(dataMatrix)#返回dataMatrix的大小，m为函数， n为列数
    alpha = 0.001#移动步长， 也就是学习速率，控制更新的幅度
    maxCycles = 500#最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)           #梯度上升矢量化公式
        error = labelMat -  h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()#将矩阵转换为数组，返回权重数组




"""
#绘图
def plotDataSet():
    dataMat, labelMat = loadDataSet()#加载数据
    dataArr = np.array(dataMat)#转换为numpy的array数据
    n = np.shape(dataMat)[0]#数据个数
    xcord1 = []; ycord1 = []  #正样本
    xcord2 = []; ycord2 = [] #负样本
    for i in range(n):#根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])        # 1为正样本
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])        #  2为负样本

        fig = plt.figure()
        ax = fig.add_subplot(111)#添加subplot
        ax.scatter(xcord1, ycord1, s=20, c='red', marker = 's', alpha=.5)#绘制正样本
        ax.scatter(xcord2, ycord2, s=20, c='green',alpha=.5)#绘制负样本
        plt.title('DataSet')#绘制标题
        plt.xlabel('X')#绘制X轴
        plt.ylabel('Y')#绘制Y轴
        plt.show()#显示图像
"""


"""
三、绘制决策边界
"""
def plotBaseFit(weights):
    dataMat, labelMat = loadDataSet()#加载数据
    dataArr = np.array(dataMat)#转换成numpy的array数组
    n = np.shape(dataMat)[0]#数据个数
    xcord1 = []; ycord1 = []#正样本
    xcord2 = []; ycord2 = []#负样本
    for i in range(n):#根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i, 2])# 1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i, 2])# 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)#添加subplot
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=.5)#绘制正样本
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)#绘制负样本
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    print(gradAscent(dataMat, labelMat))
    plotBaseFit(weights)



