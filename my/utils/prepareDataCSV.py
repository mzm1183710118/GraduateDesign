import numpy as np
import pandas as pd
import torch
from torch.utils import data


def prepare_x(data, num_features=40):

    df1 = data.iloc[:,:40]
    # 指明num_features为42时，则使用新增的2个特征
    if num_features==42:
        df2 = data.iloc[:,-2:]
        df1[['OIstd','OIRstd']] = df2
    return np.array(df1)

def get_label(data):
    lob = data.iloc[:,-30:]
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    # N是样本点的数量，D是特征数量(这里就是40)
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX, dataY

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T, num_features=40):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data, num_features)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
     
        y = y[:,self.k]-1
        self.length = len(x)

        x = torch.from_numpy(x)
        # 给x在dim=1处增加一个维度，以作为pytorch统一输入格式
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]
    
def splitDataset(train_test_rate, train_val_rate, train7path, test7path, test8path, test9path):
    '''
    train_test_rate: 训练集和测试集划分的比例，0.8表示训练集占据0.8，剩下的0.2属于测试集
    train_val_rate: 训练集和验证集划分的比例，0.8表示训练集占据0.8，剩下的0.2属于验证集
    '''
    dec_data = pd.read_csv(train7path)
    dec_train = dec_data.iloc[:int(np.floor(dec_data.shape[0] * train_test_rate))]
    dec_val = dec_data.iloc[int(np.floor(dec_data.shape[0] * train_val_rate)):]   

    dec_test1 = pd.read_csv(test7path)
    dec_test2 = pd.read_csv(test8path)
    dec_test3 = pd.read_csv(test9path)
    
    frames = [dec_test1, dec_test2, dec_test3]
    dec_test = pd.concat(frames)
    return dec_train, dec_val, dec_test

def getDataLoader(dec_train, dec_val, dec_test, k=4, num_classes=3, T=100, batch_size=64,num_features=40):
    '''
    k代表使用第几个label
    T代表对于每个样本点，一共采集多少个时间步的特征
    '''
    dataset_train = Dataset(data=dec_train, k=k, num_classes=num_classes, T=T, num_features=num_features)
    dataset_val = Dataset(data=dec_val, k=k, num_classes=num_classes, T=T, num_features=num_features)
    dataset_test = Dataset(data=dec_test, k=k, num_classes=num_classes, T=T, num_features=num_features)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def create_y(inArray, point1, point2):
    outarray = []
    for x in inArray:
        if x<point1:
            outarray.append(0)
        elif x>point2:
            outarray.append(2)
        else:
            outarray.append(1)
    outarray = np.array(outarray)
    return outarray

def changeDF(indf,T=10):
    indf['midprice'] = (indf['0']+indf['2'])/2.0
    MP_array = np.array(indf['midprice'])
    N = len(MP_array)
    newArray = []
    for i in range(N):
        # 前面大部分元素
        if i<=N-1-T:
            tmp = (np.mean(MP_array[i:i+T])-MP_array[i])/MP_array[i]
            newArray.append(tmp)
        # 对于最后一个元素 由于其没有后继 故补充0
        elif i==N-1:
            newArray.append(0)
        # 尾部元素只需要计算剩下的全部值的均值即可
        else:
            tmp = (np.mean(MP_array[i:-1])-MP_array[i])/MP_array[i]
            newArray.append(tmp)
    indf['MPchange'] = np.array(newArray)
    # 下面开始找三分位点
    a=np.array(indf['MPchange'])
    point1, point2 = np.percentile(a,33), np.percentile(a,66)
    y_array = create_y(a, point1, point2)
    indf[f'y{T}'] = y_array
    # 可以考虑丢掉原来的5个标签
    # indf.drop(columns=['144','145','146','147','148'],inplace=True)
    return indf