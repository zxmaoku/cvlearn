import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from numpy import genfromtxt
from mpl_toolkits.mplot3d import  Axes3D
from sklearn.impute import SimpleImputer
##线性回归 是用一条直线  来拟合样本点   而多项式回归是用一条曲线来拟合这些样本点
##首先  读取文件   设置默认字符格式为utf-8
x = np.random.uniform(-3, 3, size=50)
print(x.shape)
y = 0.5  + x**2 + x +0.2*x**3+3*x**2 + np.random.normal(-3, 3, size=50)
plt.scatter(x, y)
plt.show()
###首先用一元线性回归 来拟合这条曲线
model=LinearRegression()
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
##画图
model.fit(x,y)
plt.scatter(x,y,c='b')
plt.plot(x,model.predict(x),'r')
plt.show()
####定义多项式回归   degree的值可以调解多项式的特征
##因为最高次幂是五次幂  所以需要对原数据进行特征处理  比如degree是5 就x的0次项生成到5次项  并用这些数据进行特征训练
ploy_reg=PolynomialFeatures(degree=3)
##特征处理
x_poly=ploy_reg.fit_transform(x)
##定义一个回归模型
lin_reg=LinearRegression()
##把进行特征处理的数据带入模型训练
lin_reg.fit(x_poly,y)

plt.scatter(x, y,c='b')
plt.plot(x,lin_reg.predict(ploy_reg.fit_transform(x)),c='r')
plt.show()


