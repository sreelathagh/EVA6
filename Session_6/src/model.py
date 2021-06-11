import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary



class MNIST_GroupNorm(nn.Module):
    def __init__(self):
        super(MNIST_GroupNorm, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), # 28x28x1 --> 26x26x12 --> 3x3
            nn.ReLU(),
            nn.GroupNorm(num_channels=12,num_groups=3),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # 26x26x12 --> 24x24x16 --> 5x5
            nn.ReLU(),
            nn.GroupNorm(num_channels=16,num_groups=4),
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=(1, 1), padding=1, bias=False), # 24x24x16 --> 24x24x6 --> 5x5
            nn.ReLU(),
            nn.GroupNorm(num_channels=6,num_groups=2),
            nn.MaxPool2d(2, 2) # 24x24x6 --> 12x12x6 --> 6x6
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), # 12x12x6 --> 10x10x12 --> 12x12
            nn.ReLU(),
            nn.GroupNorm(num_channels=12,num_groups=3),
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), # 10x10x12 --> 8x8x14 --> 16x16
            nn.ReLU(),
            nn.GroupNorm(num_channels=14,num_groups=2),
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), # 8x8x14 --> 6x6x14 --> 20x20
            nn.ReLU(),
            nn.GroupNorm(num_channels=14,num_groups=2),
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=1, bias=False), # 6x6x14 --> 6x6x14 --> 24x24
            nn.ReLU(),
            nn.GroupNorm(num_channels=14,num_groups=2),
        ) 
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 6x6x14 --> 1x1x14 --> 24x24
            # nn.Linear(32, 10),
        )

        self.classifier = nn.Sequential(
            # nn.Linear(16, 12),
            nn.Linear(14, 10))  # 1x1x14 --> 1x10

    def forward(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.gap(x)   
        x = x.view(-1, 14)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class MNIST_BatchNorm(nn.Module):
    def __init__(self):
        super(MNIST_BatchNorm, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), # 28x28x1 --> 26x26x12 --> 3x3
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # 26x26x12 --> 24x24x16 --> 5x5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=(1, 1), padding=1, bias=False), # 24x24x16 --> 24x24x6 --> 5x5
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2, 2) # 24x24x6 --> 12x12x6 --> 6x6
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), # 12x12x6 --> 10x10x12 --> 12x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), # 10x10x12 --> 8x8x14 --> 16x16
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), # 8x8x14 --> 6x6x14 --> 20x20
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=1, bias=False), # 6x6x14 --> 6x6x14 --> 24x24
            nn.ReLU(),
            nn.BatchNorm2d(14),
        ) 
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 6x6x14 --> 1x1x14 --> 24x24
        )

        self.classifier = nn.Sequential(
            # nn.Linear(16, 12),
            nn.Linear(14, 10))  # 1x1x14 --> 1x10

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        # x = self.convblock3(x)
        x = self.gap(x)   
        x = x.view(-1, 14)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class MNIST_LayerNorm(nn.Module):
    def __init__(self):
        super(MNIST_LayerNorm, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), # 28x28x1 --> 26x26x12 --> 3x3
            nn.LayerNorm([26],elementwise_affine=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # 26x26x12 --> 24x24x16 --> 5x5
            nn.LayerNorm([24],elementwise_affine=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=(1, 1), padding=1, bias=False), # 24x24x16 --> 24x24x6 --> 5x5
            nn.LayerNorm([26],elementwise_affine=True),
            nn.ReLU(),

            nn.MaxPool2d(2, 2) # 24x24x6 --> 12x12x6 --> 6x6
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), # 12x12x6 --> 10x10x12 --> 12x12
            nn.LayerNorm([11],elementwise_affine=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), # 10x10x12 --> 8x8x14 --> 16x16
            nn.LayerNorm([9],elementwise_affine=True),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), # 8x8x14 --> 6x6x14 --> 20x20
            nn.LayerNorm([7],elementwise_affine=True),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=1, bias=False), # 6x6x14 --> 6x6x14 --> 24x24
            nn.LayerNorm([7],elementwise_affine=True),
            nn.ReLU(),
            
        ) 
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 6x6x14 --> 1x1x14 --> 24x24
        )

        self.classifier = nn.Sequential(
            nn.Linear(14, 10))  # 1x1x14 --> 1x10

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.gap(x)   
        x = x.view(-1, 14)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class Model_loader:
    
    def models(normalise,device):
        if normalise == 'BN':
            modelBatch = MNIST_BatchNorm().to(device)
            print(summary(modelBatch,input_size=(1,28,28)))
            return modelBatch

        elif normalise == 'LN':
            modelLayer = MNIST_LayerNorm().to(device)
            print(summary(modelLayer, input_size=(1, 28, 28)))
            return modelLayer
            
        else:
            modelGroup = MNIST_GroupNorm().to(device)
            print(summary(modelGroup, input_size=(1, 28, 28)))
            return modelGroup
