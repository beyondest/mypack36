import torch
import torch.nn as nn

class MobileNetV2Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MobileNetV2Bottleneck, self).__init__()
        hidden_dim = in_channels * expansion_factor

        #expand to higher dimension
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        #depthwise convolution
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)

        #project to lower dimension        
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        #init shortcut connection to be right shape
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x + shortcut




class ResNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBottleneck, self).__init__()
        hidden_dim = out_channels // 4  # ResNet中，Bottleneck的输出通道数是输出通道数的4倍

        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 convolution
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)

        # 1x1 linear projection
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x + shortcut




