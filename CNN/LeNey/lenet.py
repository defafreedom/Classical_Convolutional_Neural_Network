# coding:utf-8
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet_model(nn.Module):
    def __init__(self):
        super(LeNet_model, self).__init__()
        self.conv = nn.Sequential(  nn.Conv2d(3,6,5),      # in_channels=1,out_channels=6,kernel_size=5
                                    nn.Sigmoid(),
                                    nn.MaxPool2d(2,2),

                                    nn.Conv2d(6,16,5),          #in_channels=6,out_channels=16,kernel_size=5
                                    nn.Sigmoid(),
                                    nn.MaxPool2d(2,2))     #  stride=2 ,kernel_size = 2

        self.fc = nn.Sequential(nn.Linear(16*4*4,120),      # 16*4*4  针对输入图像是 28* 28
                                nn.Linear(120,84),
                                nn.Linear(84,10))

    def forward(self,x):
        in_size = x.size(0)
        feature = self.conv(x)
        output = self.fc(feature.view(in_size,-1))
        return output


net = LeNet_model()
summary(net,(3,28,28),device='cpu')
# X = torch.tensor(1,1,224,224)
# X = torch.rand(1,1,28,28)
# net = LeNet_model()
# X = net(X)
# print( "output shape:\t", X.shape)
# print(X)
# print(net(X).size)
