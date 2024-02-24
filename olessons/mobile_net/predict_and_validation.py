import onnxruntime
import numpy as np
import sys

import cv2
sys.path.append('../..')
from os_op.decorator import *
from params import *
from utils_network.actions import *
from torch.utils.data import Subset


onnx_filename = './weights/old83.onnx'

single_file_path = './res/dog1.jpg'

"""
mode:
0: single input, onnx reference
1: multi input, onnx reference
2: single input, torch reference
3: multi input, torch reference

"""
mode = 3






if mode == 0:
    onnx_engine = Onnx_Engine(onnx_filename,if_offline=False)

    ori_img = cv2.imread(single_file_path)

    final_input = val_trans(ori_img)
    final_input = torch.unsqueeze(final_input,dim=0)
    final_input = final_input.numpy()

    out,t = onnx_engine.run(None,{'input':final_input})


    print(f'{t:6f}')
    p,i = trans_logits_in_batch_to_result(out[0])
    print(p,i)
    
    
elif mode ==1:
    onnx_engine = Onnx_Engine(onnx_filename,if_offline=True)

    dataset = datasets.ImageFolder(val_root_path,val_trans)
    subset = get_subset(dataset,[1000,1500])
    data_loader = DataLoader(subset,
                             1,
                             True)
    
    onnx_engine.eval_run_node0(data_loader,
                               'input')
    


elif mode == 2:
    model = mobilenet_v2(num_classes=2)

    predict_classification(model,
                       val_trans,
                       single_file_path,
                       trained_weights_path,
                       class_yaml_path,
                       if_draw_and_show_result=True)

elif mode == 3:
    
    
    model = mobilenet_v2(num_classes=2)
    validation(model,val_trans,1,val_root_path,trained_weights_path)

