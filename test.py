from autoaim_alpha.autoaim_alpha.utils_network.actions import *
from autoaim_alpha.autoaim_alpha.utils_network.mymodel import *
from autoaim_alpha.autoaim_alpha.utils_network.data import *
from autoaim_alpha.autoaim_alpha.img.tools import cvshow

import torch.cuda
import cv2



def cv_trans(img:np.ndarray):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #_,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return img

input_name = 'inputs'
output_name = 'outputs'

val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    PIL_img_transform(cv_trans,'rgb'),
    transforms.ToTensor(),
    transforms.Resize((32,32))
    # transforms.Normalize((0.1307,), (0.3081,))
])

real_time_trans = transforms.Compose([
    transforms.ToTensor()
])
          
            
wei = './weights/binary.pth'
wei2 = './weights/11multi.pth'
onnx_path = './weights/binary.onnx'
onnx_path2 = './weights/11multi.onnx'


test_path = 'roi_tmp.jpg'
class_yaml_path = './guess.yaml'


model = QNet(11)


real_time_img = cv2.imread(test_path,cv2.IMREAD_GRAYSCALE)
_,real_time_img = cv2.threshold(real_time_img,127,255,cv2.THRESH_BINARY)
real_time_img = cv2.resize(real_time_img,(32,32))



if 0:

    predict_classification(model=model,
                        trans=val_trans,
                        img_or_folder_path=test_path,
                        weights_path=wei2,
                        class_yaml_path=class_yaml_path,
                        fmt='jpg',
                        custom_trans_cv=None,
                        if_show_after_cvtrans=True,
                        if_draw_and_show_result=True
                        )
if 1:
    
    onnx_engine = Onnx_Engine(onnx_path2,True)
    inp = nomalize_for_onnx([real_time_img],dtype=np.float32)
    out_list,t = onnx_engine.run(output_nodes_name_list=None,
                    input_nodes_name_to_npvalue={input_name:inp})
    
    class_dict = Data.get_file_info_from_yaml(class_yaml_path)
    p,i = trans_logits_in_batch_to_result(out_list[0])
    print(p,i)
    result_list = [class_dict[j] for j in i]
    print('probabilities:',p)
    print('results',result_list)



    

    
    