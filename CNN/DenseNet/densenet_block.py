import torch
from torch import nn
from torch.nn import functional as F

class Dense_Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Dense_Block, self).__init__()
        self.con1x1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                              kernel_size=1))
        self.con3x3 = nn.Sequential(nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                                              kernel_size=3,stride=1,padding=1))
    def forward(self,x):
        output = self.con1x1(x)
        output = self.con3x3(output)
        output = [output,x]
        return torch.cat(output,dim=1)


class Transition_Layer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Transition_Layer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        output = self.conv(F.relu(self.bn(x)))
        output = self.avgpool(output)
        return output
