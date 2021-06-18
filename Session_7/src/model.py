import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def DepthSep2d(in_channels: int, out_channels: int, kernel_size=3, padding=1):
            """Depthwise seperable convolution - combination of Depthwise and pointwise convolutions.

            Args:
                in_channels (int): Number of input channels
                out_channels (int): Number of output channels
                kernel_size (int, optional): Kernel size to use for Depthwise convolution. Defaults to 3.
                padding (int, optional): Padding value to use for depthwise convolution. Defaults to 1.

            Returns:
                nn.Sequential: Depthwise seperable convolution operation object.
            """
            depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, padding=padding)
            point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

            return nn.Sequential(depth_conv, point_conv)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            DepthSep2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
        )

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            DepthSep2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
        )

        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.conv_block3  = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ) 
        
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.transition1(x)
        x = self.conv_block2(x)
        x = self.transition2(x)
        x = self.conv_block3(x)
        x = self.gap(x)

        x = x.reshape(-1, 10)
        return x


net = Net()


class Model_loader:

    def models(device):

        model = Net().to(device)
        print(summary(model, input_size=(3, 32, 32)))
        return model
