import random
import sys
import time

import math
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
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


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self, x):# x shape: (batch, *, *, ...)
        return x.view(x.shape[0],-1)


def load_data_jay_lyrics():
    """加载歌词数据集"""
    # 读取数据集
    with open('./Datasets/66.txt') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    # corpus_chars = corpus_chars[0:10000]
    # 建立字符索引
    # 将每个字符映射成一个从0开始的连续整数，成为索引
    idx_to_char = list(set(corpus_chars))  # 取出数据集中所有不同字符
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])  # enumerate()可用输出数据和数据下标
    vocab_size = len(char_to_idx)  # 词典中不同字符的个数，又称词典大小

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    """ 随机采样 """
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps#向下取整，得到不重叠的样本个数
    example_indices = [i*num_steps for i in list(range(num_examples))] #每个样本的第一个字符在corpus_indices中的下标
    random.shuffle(example_indices) #打乱顺序

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(0, num_examples, batch_size):
        # 每次读取batch_size个随机样本
        batch_indices = example_indices[i: i + batch_size]#当前batch的各个样本的首字符的下标
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    """ 相邻采样 """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
    corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y




def grad_clipping(params,theta,device):
    """裁剪梯度
    循环神经网络容易出现梯度衰减或梯度爆炸
    把所有模型参数梯度的元素拼接成一个向量g，并设置裁剪的阈值theta
    """
    norm=torch.tensor([0.0],device=device)
    for param in params:
        norm+=(param.grad.data**2).sum()
    norm=norm.sqrt().item()  # L2范数
    if norm>theta:
        for param in params:
            param.grad.data*=(theta/norm)#分母就是L2范数


#one-hot变量
def one_hot(x,n_class,dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x=x.long()
    res=torch.zeros(x.shape[0],n_class,dtype=dtype,device=x.device)
    res.scatter_(1,x.view(-1,1),1)
    # scatter_(input, dim, index, src)
    # 将src中数据根据index中的索引按照dim的方向填进input中.
    return res


def to_onehot(X,n_class):
    #X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:,i],n_class) for i in range(X.shape[1])]


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size) # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]] # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple): # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance (state, tuple): # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))

