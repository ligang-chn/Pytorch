import random

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
