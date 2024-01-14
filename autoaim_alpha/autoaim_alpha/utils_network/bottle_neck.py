import torch
import torch.nn 



class Mobilenet_V2_bottle_neck(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,expand_factor:float,stride:int = 1) -> None:
        super().__init__()
        
        hidden_dim = in_channels * expand_factor
        
        #expand to higher dimensions
        self.conv1 = torch.nn.Conv2d(in_channels,
                                        hidden_dim,
                                        kernel_size=1,
                                        bias=False)  
        self.bn1 = torch.nn.BatchNorm2d(hidden_dim)
        self.relu1 = torch.nn.ReLU6(inplace=True)           #inplace = True means will not generate a new tensor but just rectifiy in ori_tensor
                                                            #space-accuracy trade-off
                                                            
        #depthwise convolution
        self.conv2 = torch.nn.Conv2d(   hidden_dim,
                                        hidden_dim,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,             # padding is 1 to make sure that output has same shape in this situation
                                        groups=hidden_dim,     # groups is input to make sure that this is a depthwise convolution
                                        bias=False) 
               
        self.bn2 = torch.nn.BatchNorm2d(hidden_dim)
        self.relu2 = torch.nn.ReLU6(inplace=True)
        
        #project to lower dimensions(pointwise convolution)
        self.conv3 = torch.nn.Conv2d(hidden_dim,
                                        out_channels,
                                        kernel_size=1,
                                        bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        
        #judge if change shape for shortcut is needed
        if stride != 1 or in_channels!= out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Sequential()
            
    def forward(self,x):
        shortcut = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        return shortcut+x
    

    
        

class ResNetBottleneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBottleneck, self).__init__()
        hidden_dim = out_channels // 4  # ResNet中，Bottleneck的输出通道数是输出通道数的4倍

        # 1x1 convolution
        self.conv1 = torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(hidden_dim)
        self.relu1 = torch.nn.ReLU(inplace=True)

        # 3x3 convolution
        self.conv2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(hidden_dim)
        self.relu2 = torch.nn.ReLU(inplace=True)

        # 1x1 linear projection
        self.conv3 = torch.nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)

        
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x + shortcut




class QBasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels * expansion, kernel_size=3, stride=1,
                      padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels * expansion),
            torch.nn.Conv2d(in_channels=out_channels * expansion, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            torch.nn.ReLU6()
        )

    def forward(self, x):
        return self.conv(x.clone()) + x



