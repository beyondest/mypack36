import onnxruntime
import numpy as np
import sys

import cv2
sys.path.append('../..')
from os_op.decorator import *
from params import *
from utils_network.actions import *

onnx_filename = 'best71.onnx'


ori_img = cv2.imread('3.png')





#
img = write_img_trans(ori_img)
input_tensor = val_trans(img)
input_tensor = torch.unsqueeze(input_tensor,0)
final_input= input_tensor.numpy()
#
onnx_engine = Onnx_Engine(onnx_filename)




out,t = onnx_engine.run(None,{'input':final_input})
print(f'{t:6f}')
p,i = trans_logits_in_batch_to_result(out[0])
print(p,i)

