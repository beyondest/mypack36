
import cv2

from autoaim_alpha.autoaim_alpha.utils_network.data import *
from autoaim_alpha.autoaim_alpha.utils_network.actions import *    

  

img_path = './roi_tmp.jpg'
trt_path = './tmp_net_config/long.trt'
class_yaml = './tmp_net_config/class.yaml'


cur_batch_size = 2
input_shape = (1,32,32)

img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
_,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
img = cv2.resize(img,(32,32))


real_time_input = normalize_to_nparray([img,img],np.float32)
class_info = Data.get_file_info_from_yaml(class_yaml)

Data.show_nplike_info([real_time_input])


engine = TRT_Engine_2(trt_path,
                    max_batchsize=5
                    )
                
out,t = engine.run({0:real_time_input})

logits = out[0].reshape(-1,len(class_info))

p_list,index_list =trans_logits_in_batch_to_result(logits)
result_list = [class_info[i] for i in index_list]

print(p_list,result_list)
print('spent time:')
print(t)



