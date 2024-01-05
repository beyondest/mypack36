import torch.onnx
import torch
import torch.onnx

import sys
sys.path.append('../..')
from utils_network.mymodel import *
from params import *
from utils_network.data import *

model = mobilenet_v2(num_classes=2)
dummy = torch.randn((1,3,224,224))


if 1:
    Data.save_model_to_onnx(model,
                            ori_onnx_path,
                            dummy,
                            trained_weights_path,
                            if_dynamic_batch_size=False,
                            opt_version=12)
    
if 0:
    Data.save_model_to_onnx(model,
                            ori_onnx_path,
                            dummy_input=dummy,
                            trained_weights_path=trained_weights_path,
                            if_dynamic_batch_size=True,
                            opt_version=12)
    
    
if 0:
    for name, param in model.named_parameters():
        print(f'{name}: {param.dtype}')


