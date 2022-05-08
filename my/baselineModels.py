import torch.nn as nn
import torch

class LinearRegression(nn.Module):
    def __init__(self, in_features, class_num):
        super(LinearRegression, self).__init__()
        # set size
        self.embedding_net = nn.Linear(in_features=in_features, out_features=class_num, bias=True)

    def forward(self, x):
        y = self.embedding_net(x)
        return y 

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
        y = self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(x)))))
        return y 

class ConvNet(nn.Module):
    def __init__(self, in_features, class_num):
        super(ConvNet, self).__init__()
        # set size
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=[4,124],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=[1,1],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=[4,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=256,
                                              out_channels=512,
                                              kernel_size=[3,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=512),
                                    nn.ReLU(inplace=True))
        self.layer_5 = nn.Sequential(nn.Linear(in_features=512, out_features=class_num, bias=True)) 

    def forward(self, x):
        conv_feat = self.conv_1(x)
        #print(conv_feat.shape)            
        conv_feat = self.conv_2(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_3(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_4(conv_feat)
        #print(conv_feat.shape)  
        conv_feat = F.avg_pool2d(conv_feat, kernel_size=conv_feat.size()[2:])
        conv_feat = conv_feat.squeeze(-1).squeeze(-1)
        #print(conv_feat.shape) 
        #assert 0==1  
        y = self.layer_5(conv_feat)
        return y


class LstmNet(nn.Module):
    def __init__(self, in_features, seq_len, class_num):
        super(LstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len = seq_len 
        self.class_num = class_num         
        self.rnn = nn.LSTM(in_features, in_features, num_layers=1)
        self.relu= nn.ReLU(inplace=False) 
        self.classifier = nn.Sequential(nn.Linear(in_features=in_features*seq_len, out_features=class_num, bias=True)) 

    def forward(self, x):
        x_reshaped = x.permute(1,0,2)
        batch_size = x_reshaped.size()[1]          
        h0 = torch.zeros(1, batch_size, self.in_features).to(x.device)
        c0 = torch.zeros(1, batch_size, self.in_features).to(x.device)
        #print(x_reshaped.shape,x_reshaped)        
        output, (hn,cn) = self.rnn(x_reshaped,(h0,c0))
        output = output.permute(1,0,2).contiguous()
        b,l,c = output.shape
        output = output.view(b,l*c)        
        y = self.classifier(self.relu(output))
        return y