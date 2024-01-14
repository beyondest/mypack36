import torch
import torch.nn
from typing import Union
from .bottle_neck import *

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch



class simple_2classification(torch.nn.Module):
    def __init__(self,
                 feature_nums:int=2,
                 feature_expand:bool=False,
                 correlation:Union[list,None]=None,
                 *args, 
                 **kwargs) -> None:
        '''
        Just one fullconnection Linear layer,\n
        if feature_expand, must input data after feature_expanded\n
        correlation: [[[input_dim...],[output_dim...],func0],[input_dim...],[output_dim...],func1]\n
        '''
        super().__init__(*args, **kwargs)
        self.fc=torch.nn.Linear(feature_nums,1)
        self.sigmoid = torch.nn.Sigmoid()
        
        self.feature_expand=feature_expand
        self.correlation=correlation
        
        
    def forward(self,x:torch.Tensor):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class lenet5(torch.nn.Module):
    """lenet5 just for test \n
    Input:
        28*28 grayscale
    
    """
    def __init__(self, 
                 class_nums:int = 10,
                 dropout:float = 0.3) -> None:
        super().__init__()
        
        self.dropout = dropout
        #layer 1
        
        self.conv1 = torch.nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2,bias=True)
        self.act1 = torch.nn.ReLU()              #tanh ,sigmoid is both ok ???
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2,stride=2,padding=0)        #Avg = average
        
        #layer2
        self.conv2 = torch.nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0,bias=True)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        
        #layer3
        
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(16*5*5,120,bias=True)
        self.fc1_act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(120,84,bias=True)
        self.fc2_act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(self.dropout)
        self.fc3 = torch.nn.Linear(84,class_nums,bias=True)    
                                           
        
        self.weights_init()
        
        
        
    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1_act(self.fc1(x))
        x = self.fc2_act(self.fc2(x))
        x = self.fc3(self.dropout(x))
        return x

    def weights_init(self):
        for m in self.modules():
            if isinstance(m,torch.nn.Conv2d):
                #when using relu as activation, kaiming is necessary to init it
                #conv is often set to fan_out, when linear is often set to fan_in
                torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias,0)
            elif isinstance(m,torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias,0)
            elif isinstance(m,torch.nn.Linear):
                #xavier makes forward and back_propergation runs fluently 
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias,0)
                      
    
        


class mobilenet_v2(torch.nn.Module):
    def __init__(self,
                 num_classes:int = 1000,
                 channel_width_alpha:float = 1, 
                 channel_multiply_factor:int = 8,
                 drop_out:float = 0.2
                 ) -> None:
        """input is 3 * 244 * 244

        Args:
            num_classes (int, optional): _description_. Defaults to 1000.
            channel_width_alpha (float, optional): _description_. Defaults to 1.
            channel_multiply_factor (int, optional): _description_. Defaults to 8.
            drop_out (float, optional): _description_. Defaults to 0.2.
        """
        super().__init__()
        
        self.alpha = channel_width_alpha
        self.factor = channel_multiply_factor
        self.first_channels = _make_divisible(32 * channel_width_alpha, channel_multiply_factor)
        self.last_channels = _make_divisible(1280 * channel_width_alpha,channel_multiply_factor)
        self.dropout = drop_out
        
        
        self.inversed_residual_setting = [
            #expand_factor, channels, nums_of_block, stride
            [1,            16,      1,             1],
            [6,            24,      2,             2],
            [6,            32,      3,             2],
            [6,            64,      4,             2],
            [6,            96,      3,             1],
            [6,            160,     3,             2],
            [6,            320,     1,             1]
        ]
        self.conv1 = torch.nn.Conv2d(3,
                                     self.first_channels,
                                     kernel_size=3,
                                     stride=2
                                     )
        
        self.bn1 = torch.nn.BatchNorm2d(self.first_channels)
        self.relu1 = torch.nn.ReLU6(inplace=True)   
        
        output_channels = self.set_inverted_residual()
        
        self.conv2 = torch.nn.Conv2d(output_channels,
                                     self.last_channels,
                                     1)
        self.bn2 = torch.nn.BatchNorm2d(self.last_channels)
        self.relu2 = torch.nn.ReLU6(inplace=True)
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.last_channels,num_classes)
        )
        self.weights_init()
        
        
    def forward(self,x):
        
        x = self.conv1(x)
        
        x = self.inverted_residual(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        
        x = self.avgpool(x)
        x = self.classifier(x)  
        return x         
        
        
    def set_inverted_residual(self)->int:
        """Set inverted_residual part, and return its output_channels

        Returns:
            int: _description_
        """
        inverted_residual = []
        input_channel = self.first_channels
        for t,c,n,s  in self.inversed_residual_setting:
            output_channel = _make_divisible(c*self.alpha,self.factor)
            for i in range(n):
                stride = s if i ==0 else 1
                inverted_residual.append(Mobilenet_V2_bottle_neck(input_channel,output_channel,t,stride))
                input_channel = output_channel
        self.inverted_residual = torch.nn.Sequential(*inverted_residual)
        
        return output_channel
    
    def weights_init(self):
        
        for m in self.modules():
            if isinstance(m,torch.nn.Conv2d):
                #when using relu as activation, kaiming is necessary to init it
                #conv is often set to fan_out, when linear is often set to fan_in
                torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias,0)
            elif isinstance(m,torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias,0)
            elif isinstance(m,torch.nn.Linear):
                
                #xavier makes forward and back_propergation runs fluently 
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias,0)
                      
    
    
            


class QNet(torch.nn.Module):
    '''
        Designed by QPC, aimed at a binary (32*32) input, and 2 classes classification.
        This is extraodinarily small so that it's necessary to design a special net.
    '''

    def __init__(self, num_classes=2):
        super().__init__()
        self.inblock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.block1 = QBasicBlock(in_channels=16, out_channels=16, expansion=4)
        self.down_sample1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.block2 = QBasicBlock(in_channels=32, out_channels=32, expansion=4)
        self.down_sample2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.down_sample3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=4, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.down_sample4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.outblock = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.inblock(x)
        x = self.block1(x)
        x = self.down_sample1(x)
        x = self.block2(x)
        x = self.down_sample2(x)
        x = self.down_sample3(x)
        x = self.down_sample4(x)
        x = x.view(-1, 128)
        x = self.outblock(x)
        return x

             
        
    
