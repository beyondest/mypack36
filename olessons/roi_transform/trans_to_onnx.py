import torch.onnx
import torch
import torch.onnx

import sys
sys.path.append('../..')
from autoaim_alpha.autoaim_alpha.utils_network.mymodel import *
from params import *
from autoaim_alpha.autoaim_alpha.utils_network.data import *

model = QNet(num_classes=11)
dummy = torch.randn((1,1,32,32))
ori_onnx_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/olessons/roi_binary/local_weights/94.4.onnx'
trained_weights_path = './local_weights/weights.0.94.4.pth'


if 0:
    Data.save_model_to_onnx(model,
                            ori_onnx_path,
                            dummy,
                            trained_weights_path,
                            if_dynamic_batch_size=False,
                            opt_version=12)
    
if 1:
    Data.save_model_to_onnx(model,
                            ori_onnx_path,
                            dummy_input=dummy,
                            trained_weights_path=trained_weights_path,
                            if_dynamic_batch_size=True,
                            opt_version=12)
    
    
if 0:
    for name, param in model.named_parameters():
        print(f'{name}: {param.dtype}')


