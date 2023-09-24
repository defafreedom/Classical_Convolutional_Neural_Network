# coding:utf-8
import torch
from torch import nn

class VGG_block(nn.Module):
    def __init__(self,num_conv,num_channels):
        super(VGG_block, self).__init__()
        blk = nn.Sequential()
        for _ in range(num_conv):
            blk.add_module(nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=3,padding=1),
                           nn.ReLU())
        blk.add_module(nn.MaxPool2d(kernel_size=2,stride=2))
        return blk
