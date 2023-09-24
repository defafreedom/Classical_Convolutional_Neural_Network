# coding:utf-8
import torch
from vgg_block import VGG_block
from torch import nn
from torchsummary import summary


# VGG-16
class VGG_16Model(nn.Module):
    def __init__(self):
        super(VGG_16Model, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.conv_2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.conv_3 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.conv_4 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.conv_5 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        # self.conv = nn.Sequential(self.conv_1,
        #                           self.conv_2,
        #                           self.conv_3,
        #                           self.conv_4,
        #                           self.conv_5)

        self.fc = nn.Sequential(nn.Linear(512*7*7,4096),
                                nn.ReLU(),
                                nn.Dropout(0.5,0.5),

                                nn.Linear(4096,4096),
                                nn.ReLU(),
                                nn.Dropout(0.5, 0.5),

                                nn.Linear(4096,1000))
    def forward(self,x):
        in_size = x.size(0)
        feature = self.conv_1(x)
        feature = self.conv_2(feature)
        feature = self.conv_3(feature)
        feature = self.conv_4(feature)
        feature = self.conv_5(feature)
        feature = feature.view(in_size,-1)
        output = self.fc(feature)
        return output



net = VGG_16Model()
summary(net,(3,224,224),device='cpu')
# X = torch.tensor(1,1,224,224)







