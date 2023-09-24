# coding:utf-8
from torchsummary import summary
from torch import nn

class NiN_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(NiN_block, self).__init__()
        self.nin_block = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1),
                                        nn.ReLU(),
                                       nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1),
                                        nn.ReLU())
    def forward(self,x):
        output = self.nin_block(x)
        return output



class NiN_model(nn.Module):
    def __init__(self):
        super(NiN_model, self).__init__()

        self.nin_block1 = NiN_block(3,96,11,4,0)

        self.Man_pool1 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.nin_block2 = NiN_block(96,256,5,1,2)

        self.Man_pool2 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.nin_block3 = NiN_block(256,384,3,1,1)

        self.Man_pool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.dropout1 = nn.Dropout(0.5,0.5)

        self.nin_block4 = NiN_block(384,10,3,1,1)

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
