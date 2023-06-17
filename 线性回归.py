import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
x_value=[i for i in range(10)]
x_train=np.array(x_value,dtype=np.float32)##
x_train=x_train.reshape(-1,1)###把数据转换成矩阵形式

y_values=[i*2+1for i in x_value]
y_train=np.array(y_values,dtype=np.float32)##
y_train=y_train.reshape(-1,1)###把数据转换成矩阵形式

class   lineregressionModel(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(lineregressionModel, self).__init__()
        self.linear=nn.Linear(input_dim,out_dim)##全连接层

    def  forward(self,x):##前向传播
        out= self.linear(x)
        return  out
in_dim=1
out_dim=1
model=lineregressionModel(in_dim,out_dim)
#指定好参数和损失模型
epochs=1000#训练次数1000次
learn_rate=0.01#xuexilv
optimizer=torch.optim.SGD(model.parameters(),lr=learn_rate)##优化器
criterion=nn.MSELoss()##损失函数
losss=[]
#训练模型
for  epoch  in range(epochs):
    epoch += 1
    #注意把数据转换成tensor格式
    inputs=torch.from_numpy(x_train)
    labels=torch.from_numpy(y_train)
    #每一次迭代  梯度都要清零
    optimizer.zero_grad()
    #前向传播
    outputs=model(inputs)
    print("outputs:"+str(outputs))
    #计算损失
    loss=criterion(outputs,labels)
    print("loss:"+str(loss.item()))
    losss.append(str(loss.item()))
    #  逆向传播
    loss.backward()
    #更新授权重参数
    optimizer.step()
    # if epoch % 50 == 0:
    #     print('epoch{},loss{}'.format(epoch,loss.item()))
data  =[float(d)  for d in losss]
x=range(1,len(data)+1)
###模型的保存  与读取
# torch.save(model.state_dict(),'model.pkl')
# model.load_state_dict(torch.load('model.pkl'))
plt.plot(x,data)
plt.xlabel('TIMES')
plt.ylabel('loss')
plt.ylim(0, 1)
plt.title('loss-line-chart')
plt.show()