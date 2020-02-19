import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data=pd.read_csv('./Datasets/house_prices/train.csv')
test_data=pd.read_csv('./Datasets/house_prices/test.csv')

# print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])

#将所有的训练数据和测试数据的79个特征按行连结
#第一个特征是id，很难扩展到测试样本，所以删除
all_features=pd.concat((train_data.iloc[:,1:-1],
                        test_data.iloc[:,1:]))

# print(all_features.shape)

#预处理
#1）对连续数值的特征做标准化：设该特征在整个数据集上的均值和标准差，
#   那么可以将该特征的每个值先减去均值再除以标准差，得到标准化后的每个特征值。
#2）对于缺失的特征值，将其替换成该特征的均值
# print(all_features.dtypes)
#首先判断出所有不是object类型的特征列，获取他们的列名
numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
all_features[numeric_features]=all_features[numeric_features].apply(
    lambda x:(x-x.mean())/(x.std()))
#标准化后，每个数值特征的均值变为0，所有可以直接用0来替换缺失值
all_features[numeric_features]=all_features[numeric_features].fillna(0)

#3）将离散数值转换成指示特征
#dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features=pd.get_dummies(all_features,dummy_na=True)


#将训练集和测试集分割开，前面合并是为了统一对数据进行处理
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)



#训练模型
#使用基本的线性回归模型和平方损失函数
loss=torch.nn.MSELoss()

def get_net(feature_num):
    """
    获取网络
    :param feature_num:特征数
    :return: 返回网络
    """
    net=nn.Linear(feature_num,1)#input:特征数，output:1,表示输出只有一个值
    for param in net.parameters():
        nn.init.normal_(param,mean=0,std=0.01)#模型参数初始化
    return net


#定义对数均方根误差
def log_rmse(net,features,labels):
    with torch.no_grad():
        #将小于1的值设置成1，使得取对数时，数值更稳定
        clipped_preds=torch.max(net(features),torch.tensor(1.0))
        #下面求平方根里有个2，是因为MSELoss中乘了1/2
        rmse=torch.sqrt(2*loss(clipped_preds.log(),labels.log()).mean())
    return rmse.item()


#优化算法
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls,test_ls=[],[]
    dataset=torch.utils.data.TensorDataset(train_features,train_labels)
    train_iter=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)
    #这里使用Adam优化算法
    optimizer=torch.optim.Adam(params=net.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
    net=net.float()
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l=loss(net(X.float()),y.float())#计算损失值
            optimizer.zero_grad()#梯度清零
            l.backward()
            optimizer.step()#迭代
        #记录每一个epoch迭代得到的参数模型的对数损失值
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls


#K折交叉验证
def get_k_fold_data(k,i,X,y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k  #确定每一折的样本数量
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)#slice函数用于切片
        X_part, y_part = X[idx, :], y[idx] #每一折的样本集
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, valid_ls = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                 range(1, num_epochs + 1), valid_ls,
                 ['train', 'valid']
                 )
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)


if __name__=="__main__":
    # pass
    k, num_epochs, lr, weight_decay, batch_size = 3, 200, 2.5, 0, 64
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    # print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
    # d2l.plt.show()
    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
    d2l.plt.show()






