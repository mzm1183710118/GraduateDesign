import torch.nn as nn
import torch
import torch.nn.functional as F

class LinearRegression(nn.Module):
    def __init__(self, in_features, class_num):
        super(LinearRegression, self).__init__()
        # set size
        self.embedding_net = nn.Linear(in_features=in_features, out_features=class_num, bias=True)

    def forward(self, x):
        # 将x拉平
        x = x.flatten(1)
        x = self.embedding_net(x)
        forecast_y = torch.softmax(x, dim=1)
        return forecast_y

class MLP(nn.Module):
    def __init__(self, in_features, class_num):
        super(MLP, self).__init__()
        # set size
        self.layer_1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=512, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_2 = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_3 = nn.Sequential(nn.Linear(in_features=1024, out_features=1024, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_4 = nn.Sequential(nn.Linear(in_features=1024, out_features=64, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_5 = nn.Sequential(nn.Linear(in_features=64, out_features=class_num, bias=True))

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(x)))))
        forecast_y = torch.softmax(x, dim=1)
        return forecast_y

class ConvNet(nn.Module):
    def __init__(self, y_len):
        super(ConvNet, self).__init__()
        self.y_len = y_len
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.fc1 = nn.Linear(2912, self.y_len)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)       
        x = x.flatten(1)
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)
        return forecast_y


class LstmNet(nn.Module):
    def __init__(self, y_len, device, hidden_size=64):
        self.device = device
        self.hidden_size = hidden_size
        super(LstmNet, self).__init__()
        self.conv1 = nn.Sequential( nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
                                    nn.LeakyReLU(negative_slope=0.01),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
                                    nn.LeakyReLU(negative_slope=0.01),
                                    nn.BatchNorm2d(32),
                                    )
        self.conv2 = nn.Sequential(
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
                                    nn.Tanh(),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
                                    nn.Tanh(),
                                    nn.BatchNorm2d(32),
                                )
        self.conv3 = nn.Sequential( nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
                                    nn.LeakyReLU(negative_slope=0.01),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
                                    nn.LeakyReLU(negative_slope=0.01),
                                    nn.BatchNorm2d(32),
                                    )
        # set size
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, y_len)     

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
  
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)
        return forecast_y