import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:,self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

def splitDataset(train_test_rate, train_val_rate, trian7path, test7path, test8path, test9path):
    '''
    train_test_rate: 训练集和测试集划分的比例，0.8表示训练集占据0.8，剩下的0.2属于测试集
    train_val_rate: 训练集和验证集划分的比例，0.8表示训练集占据0.8，剩下的0.2属于验证集
    '''
    dec_data = np.loadtxt(trian7path)
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * train_test_rate))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * train_val_rate)):]   

    dec_test1 = np.loadtxt(test7path)
    dec_test2 = np.loadtxt(test8path)
    dec_test3 = np.loadtxt(test9path)
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    return dec_train, dec_val, dec_test

def getDataLoader(dec_train, dec_val, dec_test, k=4, num_classes=3, T=100, batch_size=64):
    '''
    k代表使用第几个label
    T代表对于每个样本点，一共采集多少个时间步的特征
    '''
    dataset_train = Dataset(data=dec_train, k=k, num_classes=num_classes, T=T)
    dataset_val = Dataset(data=dec_val, k=k, num_classes=num_classes, T=T)
    dataset_test = Dataset(data=dec_test, k=k, num_classes=num_classes, T=T)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader