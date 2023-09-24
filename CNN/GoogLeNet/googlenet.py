# coding:utf-8
import torch
from torchsummary import summary
from torch import nn
from torch.nn import functional as F

class Inception_block(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception_block, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels=in_channels,out_channels=c1,kernel_size=1)

        self.conv3x3=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=c2[0],kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=c2[0],out_channels=c2[1],kernel_size=3,padding=1),
                                   nn.ReLU())

        self.conv5x5=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=c3[0],kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=c3[0],out_channels=c3[1],kernel_size=5,stride=1,padding=2),
                                   nn.ReLU())

        self.pool3x3=nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                                   nn.Conv2d(in_channels=in_channels,out_channels=c4,kernel_size=1),
                                   nn.ReLU())

    def forward(self,x):
        output1 = F.relu(self.conv1x1(x))
        output2 = self.conv3x3(x)
        output3 = self.conv5x5(x)
        output4 = self.pool3x3(x)
        output = [output1,output2,output3,output4]
        return torch.cat(output,dim=1)

class GoogLeNet_Model(nn.Module):
    def __init__(self):
        super(GoogLeNet_Model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(192),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.inceptionA = nn.Sequential(Inception_block(192,64,(96,128),(16,32),32),
                                        Inception_block(256,128,(128,192),(32,96),64),
                                        nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.inceptionB = nn.Sequential(Inception_block(480,192,(96,208),(16,48),64),
                                        Inception_block(512,160,(112,224),(24,64),64),
                                        Inception_block(512,128,(128,256),(24,64),64),
                                        Inception_block(512,112,(144,288),(32,64),64),
                                        Inception_block(528,256,(160,320),(32,128),128),
                                        nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.inceptionC = nn.Sequential(Inception_block(832,256,(160,320),(32,128),128),
                                        Inception_block(832,384,(192,384),(48,128),128),
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Dropout(0.4))

        self.fc = nn.Sequential(nn.Linear(1024,1000))

    def forward(self,x):
        in_size = x.size(0)
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.inceptionA(output)
        output = self.inceptionB(output)
        output = self.inceptionC(output)
        output = output.view(in_size,-1)
        output = self.fc(output)
        return output

net = GoogLeNet_Model()
summary(net,(3,224,224),device='cpu')