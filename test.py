# import torch
# import torch.nn as nn
# import numpy as np
# import sys
# sys.path.append("..")
# import d2lzh_pytorch as d2l
#
#
# n_train,n_test,true_w,true_b=100,100,[1.2,-3.4,5.6],5
# features=torch.randn((n_train+n_test,1))
# #print(features.shape )
# ploy_features=torch.cat((features,torch.pow(features,2),
#                          torch.pow(features,3)),1)
# labels=(true_w[0]*ploy_features[:,0]+true_w[1]+ploy_features[:,1]
#         +true_w[2]*ploy_features[:,2]+true_b)
# labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),
#                      dtype=torch.float)
#
# num_epochs,loss=100,torch.nn.MSELoss()
#
# def fit_and_plot(train_features,test_features,
#                  train_labels,test_labels):
#     net=torch.nn.Linear(train_features.shape[-1],1)
#
#     batch_size=min(10,train_labels.shape[0])
#     dataset=torch.utils.data.TensorDataset(train_features,train_labels)
#     train_iter=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)
#
#     optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
#     train_ls,test_ls=[],[]
#     for _ in range(num_epochs):
#         for X,y in train_iter:
#             l=loss(net(X),y.view(-1,1))
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#         train_labels=train_labels.view(-1,1)
#         test_labels=test_labels.view(-1,1)
#         train_ls.append(loss(net(train_features),train_labels).item())
#         test_ls.append(loss(net(test_features),test_labels).item())
#     print('final epoch:train loss',train_ls[-1],'test loss',test_ls[-1])
#     d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',
#                  range(1,num_epochs+1),test_ls,['train','test'])
#     d2l.plt.show()
#     print('weight:', net.weight.data,
#           '\nbias:',net.bias.data)



#########################################################################
#权重衰减法
##########################################################################
#高维线性回归实验
# import torch
# import torch.nn as nn
# import numpy as np
# import sys
# sys.path.append("..")
# import d2lzh_pytorch as d2l
#
# n_train, n_test, num_inputs = 20, 100, 200
# true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
#
# features = torch.randn((n_train + n_test, num_inputs))
# labels = torch.matmul(features, true_w) + true_b
# labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# train_features, test_features = features[:n_train, :], features[n_train:, :]
# train_labels, test_labels = labels[:n_train], labels[n_train:]
#
# #初始化模型参数
# def init_params():
#     w=torch.randn((num_inputs,1),requires_grad=True)
#     b=torch.zeros(1,requires_grad=True)
#     return w,b
#
#
# #定义L2范数惩罚项
# def l2_penalty(w):
#     return (w**2).sum()/2
#
#
# #定义训练和测试
# batch_size,num_epochs,lr=1,100,0.003
# net,loss=d2l.linreg,d2l.squared_loss
#
# dataset=torch.utils.data.TensorDataset(train_features,train_labels)
# train_iter=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)
#
# def fit_and_plot(lambd):
#     w,b=init_params()
#     train_ls,test_ls=[],[]
#     for _ in range(num_epochs):
#         for X,y in train_iter:
#             #添加L2范数惩罚项
#             l=loss(net(X,w,b),y)+lambd*l2_penalty(w)
#             l=l.sum()
#
#             if w.grad is not None:
#                 w.grad.data.zero_()
#                 b.grad.data.zero_()
#             l.backward()
#             d2l.sgd([w, b], lr, batch_size)
#         train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
#         test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
#     d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
#                  range(1, num_epochs + 1), test_ls, ['train', 'test'])
#
#     print('L2 norm of w:', w.norm().item())
#
#
# #简洁实现
# def fit_and_plot_pytorch(wd):
#     #对权重参数衰减，权重名称一般是以weight结尾
#     net=nn.Linear(num_inputs,1)
#     nn.init.normal_(net.weight,mean=0,std=1)
#     nn.init.normal_(net.bias,mean=0,std=1)
#     optimizer_w=torch.optim.SGD(params=[net.weight],
#                                 lr=lr,weight_decay=wd)
#     optimizer_b=torch.optim.SGD(params=[net.bias],lr=lr)
#
#     train_ls,test_ls=[],[]
#     for _ in range(num_epochs):
#         for X,y in train_iter:
#             l=loss(net(X),y).mean()
#             optimizer_w.zero_grad()
#             optimizer_b.zero_grad()
#
#             l.backward()
#
#             # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
#             optimizer_w.step()
#             optimizer_b.step()
#         train_ls.append(loss(net(train_features), train_labels).mean().item())
#         test_ls.append(loss(net(test_features), test_labels).mean().item())
#     d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
#                  range(1, num_epochs + 1), test_ls, ['train', 'test'])
#     print('L2 norm of w:', net.weight.data.norm().item())


###########################################################
#丢弃法
###########################################################
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


def dropout(X,drop_prob):
    X=X.float()
    assert 0<=1-drop_prob<=1
    keep_prob=1-drop_prob
    #这种情况下把全部元素都丢弃
    if keep_prob==0:
        return torch.zeros_like(X)
    mask=(torch.rand(X.shape)<keep_prob).float()

    return mask*X/keep_prob#拉伸


#定义模型参数
num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256

W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]


#定义模型
drop_prob1,drop_prob2=0.2,0.5

def net(X,is_training=True):
    X=X.view(-1,num_inputs)
    H1=(torch.matmul(X,W1)+b1).relu()
    if is_training:#只在训练模型时使用丢弃法
        H1=dropout(H1,drop_prob1)#在第一层全连接后添加丢弃层
    H2=(torch.matmul(H1,W2)+b2).relu()
    if is_training:
        H2=dropout(H2,drop_prob2)#在第二层全连接后添加丢弃层
    return torch.matmul(H2,W3)+b3


#简洁实现
net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1,num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2,10)
)

for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.01)






if __name__=="__main__":
    # fit_and_plot(ploy_features[:2, :], ploy_features[n_train:, :],
    #              labels[:2], labels[n_train:])
    # fit_and_plot(lambd=3)
    # d2l.plt.show()
    # fit_and_plot_pytorch(3)
    # d2l.plt.show()

    # X=torch.arange(16).view(2,8)
    # print(dropout(X, 0))
    # print(dropout(X, 0.5))
    # print(dropout(X, 1))

    # num_epochs, lr, batch_size = 5, 100.0, 256
    # loss = torch.nn.CrossEntropyLoss()
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

    num_epochs, lr, batch_size = 5, 100.0, 256
    loss = torch.nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)














