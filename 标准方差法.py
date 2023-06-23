import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from numpy import genfromtxt
from mpl_toolkits.mplot3d import  Axes3D
from sklearn.impute import SimpleImputer
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
##首先  读取文件   设置默认字符格式为utf-8
data=genfromtxt(r"./data3.csv",delimiter=';', encoding='utf-8')
###读取数据  分析每一列数据意义 1  酸度 2挥发性酸  3 柠檬酸  4 残糖  5氯化物 6游离二氧化硫  7总二氧化硫
######################     8 密度  9 ph   10硫酸盐 11 酒精 12  质量
#元素
a=data[1:,0]##酸度
b=data[1:,1]##挥发性酸
f=data[1:,2,np.newaxis]##柠檬酸
d=data[1:,8,np.newaxis]##ph
##增加偏置顶
f=np.concatenate((np.ones((1599,1)),f),axis=1)
# plt.scatter(f,d)
# plt.show()
# print(np.mat(f).shape)
#标准方程法求解回归参数
def  weight(xarry,yarry):
    xmat=np.mat(xarry)
    ymat=np.mat(yarry)
    xtx=xmat.T*xmat
    if np.linalg.det(xtx)==0.0:
        print("矩阵不可逆")
        return
    wx=xtx.I*xmat.T*ymat
    return wx
ws=weight(f,d)
xtest=np.array([[0.0],[1.0]])
ytest=ws[0]+xtest*ws[1]
plt.plot(f,d,'b')
plt.plot(xtest,ytest,'r')
plt.show()
