import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from numpy import genfromtxt
from mpl_toolkits.mplot3d import  Axes3D
from sklearn.impute import SimpleImputer
##首先  读取文件   设置默认字符格式为utf-8
data=genfromtxt(r"./data1.csv",delimiter=',', encoding='utf-8')
##数据切分
xdata=data[:,:-1]##只要前两列的数据
ydata=data[:,-1]##只要后两列的数据
##处理NaN值
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
xdata = imp.fit_transform(xdata)
ir=0##学习率
##三个参数
cs01=0
cs02=0
cs03=0
epochs=20000# 最大迭代参数

###使用sklearn  实现多元回归
mode=linear_model.LinearRegression()
mode.fit(xdata,ydata)
print("系数")
print(mode.coef_)
print("截距")
print(mode.intercept_)
#test
x_test=[[102,4]]
predict=mode.predict(x_test)
print("presiect",predict)
###画图
ax=plt.figure().add_subplot(111,projection='3d')
ax.scatter(xdata[:,0],xdata[:,1],ydata,c='r',marker='o',s=100)##红色三角形
x0=xdata[:,0]
x1=xdata[:,1]

###生成网络矩阵
x0,x1,=np.meshgrid(x0,x1)
z= mode.intercept_ + x0*mode.coef_[0] + x1*mode.coef_[1]
ax.plot_surface(x0,x1,z)###画3d图
##设置横纵坐标轴
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
##显示图像
plt.show()




##最小二乘法
# def compute_errors(cs01,cs02,cs03,xdata,ydata):##计算代价函数值
#     totalerror=0.00
#     for i   in range(0,len(xdata)):
#         # y真实值   (y[i]-(k*x[i]-b))预测值
#         totalerror += (float(ydata[i])-(cs02*float(xdata[i,0])+cs03*float(xdata[i,1])+cs01))**2
#         print(totalerror)
#         print(ydata[i])
#         print(xdata[i,0])
#         print(xdata[i,1])
#     return  totalerror/float(len(xdata))
# def gradient(xdata,ydata,cs01,cs02,cs03,ir,epochs):
#     # 计算总数数
#     m=float(len(xdata))
#     for i in  range(epochs):
#         cs01_grad=0
#         cs02_grad=0
#         cs03_grad = 0
#         ###计算各层的梯度   就是求  k b 的导数
#         for j   in range(0,len(xdata)):
#             ##获取参数导数
#             cs01_grad += -(1/m)*(ydata[j]-(cs02*xdata[j,0]+cs03_grad*xdata[j,1]+cs01))
#             cs02_grad += -(1 / m)*xdata[j,0]*(ydata[j]-(cs02*xdata[j,0]+cs03*xdata[j,1]+cs01))
#             cs02_grad += -(1 / m) * xdata[j, 1] * (ydata[j] - (cs02 * xdata[j, 0] + cs03 * xdata[j, 1] + cs01))
#             ##更新参数
#             cs01=cs01-(ir*cs01_grad)
#             cs02=cs02-(ir*cs02_grad)
#             cs03=cs03-(ir*cs03_grad)
#         # if i%500==0:
#         #     print("epochs",i)
#         #     plt.plot(x, y, 'b')
#         #     plt.plot(x, k * x + b, 'r')
#         #     plt.show()
#
#     return cs01,cs02,cs03
# print("cs01={0}  cs02={1}   cs03={2} error={3}".format(cs01,cs02,cs03,compute_errors(cs01,cs02,cs03,xdata,ydata)))
# print("running")
# cs01,cs02,cs03=gradient(xdata,ydata,cs01,cs02,cs03,ir,epochs)
# print("after{0}timescs01={1}  cs02={2}   cs03={3} error={4} ".format(epochs,cs01,cs02,cs03,compute_errors(cs01,cs02,cs03,xdata,ydata)))
