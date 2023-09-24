# coding:utf-8
import torch
from torch import nn
from torch.nn import functional as F
from resnet_block import res_block
from torchsummary import summary

class ResNet_model(nn.Module):
    def __init__(self):
        super(ResNet_model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.res_block1 = nn.Sequential(res_block(64,64,stride=1),
                                        res_block(64,64,stride=1))

        self.res_block2 = nn.Sequential(res_block(64,128,use_1x1conv=True,stride=2),
                                        res_block(128,128,stride=1))

        self.res_block3 = nn.Sequential(res_block(128,256,use_1x1conv=True,stride=2),
                                        res_block(256,256,stride=1))

        self.res_block4 = nn.Sequential(res_block(256,512,use_1x1conv=True,stride=2),
                                        res_block(512,512,stride=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,1000)

    def forward(self,x):
        in_size = x.size(0)
        output = self.conv1(x)
        output = self.res_block1(output)
        output = self.res_block2(output)
        output = self.res_block3(output)
        output = self.res_block4(output)
        output = self.avgpool(output)
        output = output.view(in_size,-1)
        output = self.fc(output)
        return output

net = ResNet_model()
summary(net,(3,224,224),device='cpu')