import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# 生成随机数据
x= np.random.rand(1000, 1)
y = np.random.rand(1000, 1)
#随机数据可能没有线性关系

# xx=np.array()
#画散点图
print(len(x))
print(len(y))
plt.scatter(x,y)
plt.show()
##学习率  影响步长
ir=0.0001
## 截距
b=0
##斜率、
k=0
##  最大迭代次数
epochs=500
# 最小二乘法
def compute_errors(b,k,x,y):##计算代价函数值
    totalerror=0
    for i   in range(0,len(x)):
        # y真实值   (y[i]-(k*x[i]-b))预测值
        totalerror +=(y[i]-(k*x[i]-b))**2
    return  totalerror/float(2*len(x))/2
allbs=[]
allks=[]
def gradient(x,y,b,k,ir,epochs):
    # 计算总数数
    m=float(len(x))
    for i in  range(epochs):
        b_grad=0
        k_grad=0
        ###计算各层的梯度   就是求  k b 的导数
        for j   in range(0,len(x)):
            b_grad += -(1/m)*(y[j]-(k*x[j]+b))
            k_grad += -(1/m)*x[j]*(y[j]-(k*x[j]+b))
            b=b-(ir*b_grad)
            k=k-(ir*k_grad)
            allbs.append(b)
            allks.append(k)
    return b,k
b ,k = gradient(x,y,b,k,ir,epochs)###迭代以后得到的bk
print("after   train{0} b={1} k={2}  error={3}".format(epochs,b,k,compute_errors(b,k,x,y)))
plt.scatter(allbs,allks)
plt.plot(x,y,'b')
plt.plot(x,k*x+b,'r')
plt.show()