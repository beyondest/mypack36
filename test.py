import os
import torch
import time
from torchvision import transforms,datasets
import cv2
import onnxruntime
from typing import Union,Optional
import torch.nn
import torch.cuda
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

except ImportError:
    lr1.error("No tensorrt or pycuda, please install tensorrt and pycuda")
    



class Trt_Engine:

    def __init__(self,
                 filename:str,
                 if_show_engine_info:bool = True
                 
                 ) -> None:
        """Config here if you wang more

        Args:
            filename (_type_): _description_
        """

        if os.path.exists(filename) == False:
            raise FileNotFoundError(f"Tensorrt engine file not found: {filename}")
        
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.trt_logger)
        
        with open(filename, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if if_show_engine_info:
            node_info = self.get_node_info()
            print(f"Engine info: {node_info}")
        print(self.engine.max_batch_size)

    def get_node_info(self):
        num_bindings = self.engine.num_bindings
        binding_info = []

        for i in range(num_bindings):
            
            binding = self.engine.binding_is_input(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            binding_info.append({
                "name": f"Binding_{i}",
                "input": binding,
                "shape": shape,
                "dtype": dtype,
            })

        return binding_info





a = Trt_Engine('./tmp_net_config/long.trt')

