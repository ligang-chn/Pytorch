import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
sys.path.append("..")
from d2lzh_pytorch import *



num_inputs=2#输入特征数
num_examples=1000#样本个数
true_w=[2,-3.4]#真实权重
true_b=4.2#偏差
#样本集
features=torch.randn(num_examples,num_inputs,dtype=torch.float32)
#标签
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
#加上噪声，噪声服从均值为0，标准差为0.01的正态分布
labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),
                     dtype=torch.float32)

#设置图片尺寸
set_figsize()

#初始化模型参数
w=torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32)
b=torch.zeros(1,dtype=torch.float32)
#需要使用这些参数求梯度来迭代参数的值
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


#定义模型
def linreg(X,w,b):
    """
    线性回归模型
    :param X: 样本集
    :param w: 参数w
    :param b: 参数b
    :return:
    """
    return torch.mm(X,w)+b#主要就是那个线性回归方程


#定义损失函数
def squared_loss(y_hat,y):
    """
    平方损失
    :param y_hat:预测值
    :param y:真实值
    :return:返回损失值
    """
    #注意这里返回的是向量
    return (y_hat-y.view(y_hat.size()))**2 / 2


#定义优化算法
def sgd(params,lr,batch_size):
    """
    小批量随机梯度下降算法
    :param params:需要迭代的参数
    :param lr: 学习率
    :param batch_size:批量大小
    :return: 迭代得到的参数
    """
    for param in params:
        param.data -= lr*param.grad/batch_size#注意更改param时使用param。data


#训练模型
lr=0.03#学习率
num_epochs=3#迭代周期个数
net=linreg#之前定义的线性模型
loss=squared_loss#之前定义的损失函数
batch_size=10

for epoch in range(num_epochs):#每一个迭代周期中，会使用训练数据集中所有样本一次，
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y).sum()#l是有关小批量X和y的损失
        l.backward()#小批量的损失对模型参数求梯度
        sgd([w,b],lr,batch_size)#使用小批量随机梯度下降法迭代模型参数

        #梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    #此时一个epoch已经结束
    train_l=loss(net(features,w,b),labels)
    print('epoch {},loss {}'.format(
        epoch+1,train_l.mean().item()))


if __name__=='__main__':
    print(true_w,'\n',w)
    print(true_b,'\n',b)
    # print("labels: {}".format(labels.shape))
    #plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    #plt.show()

    # for X,y in data_iter(batch_size,features,labels):
    #     print(X,y)
    #     break