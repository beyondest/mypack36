import torch.onnx
import torch
import torch.onnx

import sys
sys.path.append('../..')
from utils_network.mymodel import *
from params import *


model = lenet5()
dummy = torch.randn((1,1,28,28))
Data.save_model_to_onnx(model,
                        'best71.onnx',
                        dummy,
                        trained2_weights_path)

