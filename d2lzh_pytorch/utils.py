import random
import sys

import torchvision
import torchvision.transforms as transforms
import torch
from IPython import display
import matplotlib.pyplot as plt



def use_svg_display():
    """
    用矢量图显示
    :return: none
    """
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5,2.5)):
    """
    设置图的尺寸
    :param figsize: 元组表示的图尺寸大小
    :return: none
    """
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize


def data_iter(batch_size,features,labels):
    """
    产生小批量数据样本
    :param batch_size: 批量大小
    :param features: 样本集
    :param labels: 样本标签
    :return: 小批量样本集，标签
    """
    num_examples=len(features)#计算样本个数
    indices=list(range(num_examples))#生成样本个数大小的整数列表，相当于序号
    random.shuffle(indices)#将样本的顺序打乱，这样后面一次读取时，可以认为时随机的
    for i in range(0,num_examples,batch_size):#在0到num_examples中，步长为bathcn——size
        j=torch.LongTensor(indices[i:min(i+batch_size,num_examples)])# 最后一次可能不足一个batch
        yield features.index_select(0,j),labels.index_select(0,j)
        #注意这里使用的是yield，因为当一个循环结束时发送出一个批量，然后返回的到刚刚离开的位置


def get_fashion_mnist_labels(labels):
    """
    将数值标签转换成相应的文本标签
    :param labels:数值标签
    :return:文本标签列表
    """
    text_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images,labels):
    """
    绘制多张图像和对应标签
    :param images: 图像
    :param labels: 标签
    :return: 图片
    """
    use_svg_display()
    #这里的_表示我们忽略（不使用）的变量
    _,figs=plt.subplots(1,len(images),figsize=(12,12))
    for f,img,lbl in zip(figs,images,labels):
        f.imshow(img.view(28,28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()



def load_data_fashion_mnist(batch_size, root='./Datasets/FashionMNIST0216'):
    """Download the fashion mnist dataset and then load into memory."""
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    for X,y in data_iter:
        acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
    return acc_sum/n


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


def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,
              params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y).sum()

            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()#torch版本中用到

            train_l_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))