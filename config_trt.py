import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

from autoaim_alpha.autoaim_alpha.utils_network.data import *
from autoaim_alpha.autoaim_alpha.utils_network.actions import *




class HostDeviceMem(object):
    def __init__(self,host_mem,device_mem) -> None:
        self.host = host_mem
        self.device = device_mem
    def __str__(self) -> str:
        return "Host : \n" + str(self.host) + "\nDevice : \n" + str(self.device)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    


def allocate_buffers(engine,max_batch_size:int = 16,if_show:bool = True):
    """
    Return :
        list,list,list,stream_obj: inputs,outputs,bindings,stream
    """    
    assert (max_batch_size is not None)

    inputs = []
    outputs = []
    bindings = []
    stream = 0
    
    for binding in engine:
        
        dims = engine.get_binding_shape(binding)
        
        if if_show:
            print('dims:',dims)
        
        if dims[0] == -1:
            dims[0] = max_batch_size
        
        
        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        if if_show:
            print('Final:')
            print('dims : ',dims)
            print('size : ',size)
            print('dtype : ',dtype)
        
        host_mem = cuda.pagelocked_empty(size,dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem,device_mem))
        
        else:
            outputs.append(HostDeviceMem(host_mem,device_mem))
        
        
    return inputs,outputs,bindings,stream

  
            
def do_inference(context,
                 bindings:list,
                 inputs:list,
                 outputs:list,
                 stream)->list:
    """do inference by gpu

    Args:
        context (_type_): _description_
        bindings (list): _description_
        inputs (list): _description_
        outputs (list): _description_
        stream (_type_): _description_

    Returns:
        list: [out1,out2,...]
    """
   
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device,inp.host,stream)
    context.execute_async_v2(bindings =bindings,
                             stream_handle = stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host,out.device,stream)
        
    stream.synchronize()
    final_output = [out.host for out in outputs]
    return final_output




img_path = './roi_tmp.jpg'
trt_path = './tmp_net_config/long.trt'
class_yaml = './tmp_net_config/class.yaml'    

    
cur_batch_size = 2
input_shape = (1,32,32)

img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
_,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
img = cv2.resize(img,(32,32))


real_time_input = normalize_to_nparray([img,img],np.float32)


Data.show_nplike_info([real_time_input])


with open(trt_path,'rb') as f:
    serialized_engine = f.read()

logger = trt.Logger(trt.Logger.VERBOSE)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()
t1 = time.perf_counter()
stream0 = cuda.Stream()


context.set_binding_shape(0,(cur_batch_size,*input_shape))
inputs,outputs,bindings,stream = allocate_buffers(engine,
                                                  cur_batch_size,
                                                  if_show=True)

t2 = time.perf_counter()

np.copyto(inputs[0].host,real_time_input.ravel())

result = do_inference(context,
                      bindings,
                      inputs,
                      outputs,
                      stream0)

logits = result[0].reshape(cur_batch_size,-1)

t3 = time.perf_counter()


print('spend time ',t2-t1,t3 - t2)
print('total time : ',t3 - t1)


class_info = Data.get_file_info_from_yaml(class_yaml)

pro_list,index_list = trans_logits_in_batch_to_result(logits)
result_list  = [class_info[i] for i in index_list]
print(result_list)
print(pro_list)


