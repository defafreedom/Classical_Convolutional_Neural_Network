# coding:utf-8
from torch import nn
from torchsummary import summary


class AlexNet_model(nn.Module):
    def __init__(self):
        super(AlexNet_model, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3,stride=2),

                                  nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3,stride=2),

                                  nn.Conv2d(in_channels=256,out_channels=388,kernel_size=3,padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=388,out_channels=388,kernel_size=3,padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=388,out_channels=256,kernel_size=3,padding=1),
                                  nn.MaxPool2d(kernel_size=3,stride=2),
                                  )

        self.fc = nn.Sequential(nn.Linear(256*5*5,4096),   #输入图像是 224*224
                                nn.ReLU(),
                                nn.Dropout(0.5,0.5),

                                nn.Linear(4096,4096),
                                nn.ReLU(),
                                nn.Dropout(0.5, 0.5),
                                nn.Linear(4096,10)
                                )

    def forward(self,x):
        in_size = x.size(0)
        feature = self.conv(x)
        feature = feature.view(in_size,-1)
        output = self.fc(feature)
        return output

net = AlexNet_model()
summary(net,(3,224,224),device='cpu')
# X = torch.tensor(1,1,224,224)