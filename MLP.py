#多层感知机
#多层感知机在单层神经网络的基础上引入隐藏层
import torch
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l



def xyplot(x_vals,y_vals,name):
    d2l.set_figsize(figsize=(5,2.5))
    d2l.plt.plot(x_vals.detach().numpy(),y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name+'(x)')


#读取数据
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)


#定义模型参数
num_inputs=784
num_outputs=10
num_hiddens=256

W1=torch.tensor(np.random.normal(0,0.01,
                (num_inputs,num_hiddens)),
                dtype=torch.float)
b1=torch.zeros(num_hiddens,dtype=torch.float)
W2=torch.tensor(np.random.normal(0,0.01,
                 (num_hiddens,num_outputs)),
                dtype=torch.float)
b2=torch.zeros(num_outputs,dtype=torch.float)

params=[W1,b1,W2,b2]
for param in params:
    param.requires_grad_(requires_grad=True)


#定义激活函数
def relu(X):
    return torch.max(input=X,other=torch.tensor(0.0))


#定义模型
def net(X):
    X=X.view(-1,num_inputs)
    H=relu(torch.matmul(X,W1)+b1)
    return torch.matmul(H,W2)+b2


#定义损失函数
loss=torch.nn.CrossEntropyLoss()


#训练模型
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


if __name__=='__main__':
    pass
    # x=torch.arange(-8.0,8.0,0.1,requires_grad=True)
    # y=x.relu()
    # xyplot(x,y,'relu')
    # plt.show()
    # y.sum().backward()
    # xyplot(x,x.grad,'grad of relu')
    # plt.show()


























