import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2
#训练数据特征
features=torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),
                      dtype=torch.float)
#标签
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),
                                      dtype=torch.float)

#读取数据
#使用data包来读取数据
batch_size=10

#将训练数据的特征和标签组合
dataset=Data.TensorDataset(features,labels)
#随机读取小批量
data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)


####定义模型###########
#nn的核心数据结构是Module，它是一个抽象概念，
# 既可以表示神经网络中的某个层（layer），
# 也可以表示一个包含很多层的神经网络
class LinearNet(nn.Module):
    """
    一个nn.Module实例应该包含一些层
    以及返回输出的前向传播（forward）方法
    """
    def __init__(self,n_features):
        super(LinearNet,self).__init__()
        self.linear=nn.Linear(n_features,1)
    #forward定义前向传播
    def forward(self,x):
        y=self.linear(x)
        return y


net=LinearNet(num_inputs)#实例化，产生具体网络
##其他搭建网络的方式
# net=nn.Sequential(nn.Linear(num_inputs,1)) #1
# net=nn.Sequential() #2
# net.add_module('linear',nn.Linear(num_inputs,1))
# from collections import OrderedDict #3
# net = nn.Sequential(OrderedDict([
#           ('linear', nn.Linear(num_inputs, 1))
#           # ......
#         ]))


#初始化模型参数
init.normal_(net.linear.weight,mean=0,std=0.01)
init.constant_(net.linear.bias,val=0)


#定义损失函数
loss=nn.MSELoss()


#定义优化算法
optimizer=optim.SGD(net.parameters(),lr=0.03)
#为不同子网络设置不同的学习率，这在finetune时经常用到
# optimizer=optim.SGD([
#     #如果对某个参数不指定学习率，就使用最外层的默认学习率
#     {'params':net.subnet1.parameters()},#lr=0.03
#     {'params':net.subnet2.parameters(),'lr':0.03}
# ],lr=0.03)

#训练模型
#在step函数中指明批量大小，从而对批量中样本梯度求平均
num_epochs=3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output=net(X)
        l=loss(output,y.view(-1,1))
        optimizer.zero_grad()#梯度清零
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))



if __name__=='__main__':
    # print(net)
    #查看模型所有的可学习参数，此函数将返回一个生成器
    # for param in net.parameters():
    #     print(param)

    #print(net.linear)
    # for X,y in data_iter:
    #     print(X,y)
    #     break
    #print(optimizer)
    dense = net.linear
    print(true_w, dense.weight)
    print(true_b, dense.bias)











