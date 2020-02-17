from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


#读取数据
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)


#定义和初始化模型
# softmax回归的输出层是一个全连接层，所以我们用一个线性模块就可以了。
# 因为前面我们数据返回的每个batch样本x的形状为(batch_size, 1, 28, 28),
# 所以我们要先用view()将x的形状转换成(batch_size, 784)才送入全连接层。
num_inputs=784
num_outputs=10
#第一种方法
# class LinearNet(nn.Module):
#     """
#     softmax回归的线性模块
#     """
#     def __init__(self,num_inputs,num_outputs):
#         super(LinearNet,self).__init__()
#         self.linear=nn.Linear(num_inputs,num_outputs)
#
#     def forward(self, x):#x shape:(batch,1,28,28)
#         y=self.linear(x.view(x.shape[0],-1))
#         return y
#
# net=LinearNet(num_inputs,num_outputs)

#第二种方法
net=nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten',d2l.FlattenLayer()),
        ('linear',nn.Linear(num_inputs,num_outputs))
    ])
)

#初始化权重参数
init.normal_(net.linear.weight,mean=0,std=0)
init.constant_(net.linear.bias,val=0)


#交叉熵损失函数
#PyTorch提供了一个包括softmax运算和交叉熵损失计算的函数
loss=nn.CrossEntropyLoss()


#定义优化算法-小批量随机梯度下降法
optimizer=torch.optim.SGD(net.parameters(),lr=0.1)


#训练模型
num_epochs=5
d2l.train_ch3(net,train_iter,test_iter,
              loss,num_epochs,batch_size,
              None,None,optimizer)



if __name__=='__main__':
    pass
    # for i in train_iter:
    #     print(i)
    #     break



