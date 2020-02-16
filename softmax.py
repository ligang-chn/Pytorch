import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


# mnist_train=torchvision.datasets.FashionMNIST(
#     root='./Datasets/FashionMNIST0216',
#     train=True,
#     download=True,
#     transform=transforms.ToTensor()
# )
# mnist_test=torchvision.datasets.FashionMNIST(
#     root='./Datasets/FashionMNIST0216',
#     train=False,
#     download=True,
#     transform=transforms.ToTensor()
# )




#设置批量大小
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)


#初始化模型参数
num_inputs=784
num_outputs=10

W=torch.tensor(np.random.normal(0,0.01,
                                (num_inputs,num_outputs)),
               dtype=torch.float)
b=torch.zeros(num_outputs,dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


#实现softmax运算
def softmax(X):
    X_exp=X.exp()#首先对每个元素做指数运算
    partition=X_exp.sum(dim=1,keepdim=True)#对同一行数据进行求和
    return X_exp/partition #这里使用了广播机制


#定义模型
def net(X):
    return softmax(torch.mm(X.view((-1,num_inputs)),W)+b)


#定义损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat.gather(1,y.view(-1,1)))


#计算分类准确率
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1)==y).float().mean().item()


#训练模型
num_epochs,lr=5,0.1
d2l.train_ch3(net, train_iter, test_iter,
          cross_entropy, num_epochs,
          batch_size,
          [W, b], lr)

if __name__=='__main__':

    # print(type(mnist_train))
    # print(len(mnist_train), len(mnist_test))
    # print(d2l.evaluate_accuracy(test_iter,net))

    X,y=iter(test_iter).next()

    true_labels=d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels=d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    d2l.show_fashion_mnist(X[0:9], titles[0:9])

