# coding:utf-8
from densenet_block import Dense_Block
from densenet_block import Transition_Layer
import torch
from torch import nn
from torchsummary import summary

class DenseNet_model(nn.Module):
    def __init__(self):
        super(DenseNet_model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,stride=2))

        self.DB1 = nn.Sequential(Dense_Block(64,32),
                                 Dense_Block(96,32),
                                 Dense_Block(128,32),
                                 Dense_Block(160,32),
                                 Dense_Block(192, 32),
                                 Dense_Block(224, 32)
                                 )
        self.TL1 = Transition_Layer(256,128)

        self.DB2 = nn.Sequential(Dense_Block(128,16),
                                 Dense_Block(144,16),
                                 Dense_Block(160,16),
                                 Dense_Block(176,16),
                                 Dense_Block(192, 16),
                                 Dense_Block(208, 16),
                                 Dense_Block(224, 16),
                                 Dense_Block(240, 16),
                                 Dense_Block(256, 16),
                                 Dense_Block(272, 16),
                                 Dense_Block(288, 16),
                                 Dense_Block(304, 16)
                                 )

        self.TL2 = Transition_Layer(320,160)

        self.DB3 = nn.Sequential(Dense_Block(160,16),
                                 Dense_Block(176,16),
                                 Dense_Block(192,16),
                                 Dense_Block(208,16),
                                 Dense_Block(224, 16),
                                 Dense_Block(240, 16),
                                 Dense_Block(256, 16),
                                 Dense_Block(272, 16),
                                 Dense_Block(288, 16),
                                 Dense_Block(304, 16),
                                 Dense_Block(320, 16),
                                 Dense_Block(336, 16),
                                 Dense_Block(352, 16),
                                 Dense_Block(368, 16),
                                 Dense_Block(384, 16),
                                 Dense_Block(400, 16),
                                 Dense_Block(416, 16),
                                 Dense_Block(432, 16),
                                 Dense_Block(448, 16),
                                 Dense_Block(464, 16),
                                 Dense_Block(480, 16),
                                 Dense_Block(496, 16),
                                 Dense_Block(512, 16),
                                 Dense_Block(528, 16)
                                 )

        self.TL3 = Transition_Layer(544,160)

        self.DB4 = nn.Sequential(Dense_Block(160,16),
                                 Dense_Block(176,16),
                                 Dense_Block(192,16),
                                 Dense_Block(208,16),
                                 Dense_Block(224, 16),
                                 Dense_Block(240, 16),
                                 Dense_Block(256, 16),
                                 Dense_Block(272, 16),
                                 Dense_Block(288, 16),
                                 Dense_Block(304, 16),
                                 Dense_Block(320, 16),
                                 Dense_Block(336, 16),
                                 Dense_Block(352, 16),
                                 Dense_Block(368, 16),
                                 Dense_Block(384, 16),
                                 Dense_Block(400, 16))

        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(416,1000)

    def forward(self,x):
        in_size = x.size(0)
        output = self.conv1(x)
        output = self.DB1(output)
        output = self.TL1(output)
        output = self.DB2(output)
        output = self.TL2(output)
        output = self.DB3(output)
        output = self.TL3(output)
        output = self.DB4(output)
        output = self.global_avgpool(output)
        output = output.view(in_size,-1)
        output = self.fc(output)
        return output

net = DenseNet_model()
summary(net,(3,224,224),device='cpu')




