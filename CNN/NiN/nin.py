# coding:utf-8
import torch
from torchsummary import summary
from torch import nn

class NiN_model(nn.Module):
    def __init__(self):
        super(NiN_model, self).__init__()

        self.nin_block1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=96,out_channels=96,kernel_size=1),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=96,out_channels=96,kernel_size=1),
                                        nn.ReLU())

        self.Man_pool1 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.nin_block2 = nn.Sequential(nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
                                        nn.ReLU())

        self.Man_pool2 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.nin_block3 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=384,out_channels=384,kernel_size=1),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=384,out_channels=384,kernel_size=1),
                                        nn.ReLU())

        self.Man_pool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.dropout1 = nn.Dropout(0.5,0.5)

        self.nin_block4 = nn.Sequential(nn.Conv2d(in_channels=384,out_channels=10,kernel_size=3,stride=1,padding=1),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=10,out_channels=10,kernel_size=1),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=10,out_channels=10,kernel_size=1),
                                        nn.ReLU())

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.Flatten = nn.Flatten()


    def forward(self,x):
        # h,w = x.size(3),x.size(4)
        output = self.nin_block1(x)
        output = self.Man_pool1(output)
        output = self.nin_block2(output)
        output = self.Man_pool2(output)
        output = self.nin_block3(output)
        output = self.Man_pool3(output)
        output = self.dropout1(output)
        output = self.nin_block4(output)
        output = self.globalavgpool(output)
        output = self.Flatten(output)
        return output

net = NiN_model()
summary(net,(3,224,224),device='cpu')

