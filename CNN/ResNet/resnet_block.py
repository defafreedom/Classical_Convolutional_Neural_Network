# coding:utf-8
import torch
from torch import nn
from torch.nn import functional as F

class res_block(nn.Module):
    def __init__(self,in_channels,out_channels,use_1x1conv=False,stride=1):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                               kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                               kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=1,stride=stride)
        else:
            self.conv3 = None

    def forward(self,x):
        output1 = F.relu(self.bn1(self.conv1(x)))
        output1 = self.bn2(self.conv2(output1))
        if self.conv3:
            output2 = self.conv3(x)
            output = F.relu(output1 + output2)
        else:
            output = output1
        return output



