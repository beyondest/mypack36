from autoaim_alpha.autoaim_alpha.utils_network.actions import *
from autoaim_alpha.autoaim_alpha.utils_network.mymodel import *
from autoaim_alpha.autoaim_alpha.utils_network.data import *
from autoaim_alpha.autoaim_alpha.img.tools import *
from autoaim_alpha.autoaim_alpha.utils_network.api_for_yolov5 import *
import torch.cuda
import cv2
import torch
import torchvision  

def cv_trans(img:np.ndarray):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #_,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return img


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
binary_onnx_path = './weights/11multi.onnx'
yolo_onnx_path = './weights/batchsize1_onnx/ori.onnx'

test_path = 'roi_tmp.jpg'
test_path2 = 'armorred.png'
class_yaml_path = './tmp_net_config/class.yaml'
yolo_class_yaml_path = './yolov5_class.yaml'



binary_pth_model = False
binary_onnx_model = False
yolo_pth_model = False
yolo_onnx_model = True


if binary_pth_model:
    
    model = QNet(11)
    
    real_time_img = cv2.imread(test_path,cv2.IMREAD_GRAYSCALE)
    _,real_time_img = cv2.threshold(real_time_img,127,255,cv2.THRESH_BINARY)
    real_time_img = cv2.resize(real_time_img,(32,32))

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
    
if binary_onnx_model:
    
    input_name = 'input'
    output_name = 'output'

    real_time_img = cv2.imread(test_path,cv2.IMREAD_GRAYSCALE)
    _,real_time_img = cv2.threshold(real_time_img,127,255,cv2.THRESH_BINARY)
    real_time_img = cv2.resize(real_time_img,(32,32))

    onnx_engine = Onnx_Engine(binary_onnx_path,True)
    
    inp = normalize_to_nparray([real_time_img],dtype=np.float32)
    
    out_list,t = onnx_engine.run(output_nodes_name_list=None,
                    input_nodes_name_to_npvalue={input_name:inp})
    
    class_dict = Data.get_file_info_from_yaml(class_yaml_path)
    print(out_list[0].shape)
    p,i = trans_logits_in_batch_to_result(out_list[0])
    
    print(p,i)
    result_list = [class_dict[j] for j in i]
    print('probabilities:',p)
    print('results',result_list)
    
    print('onnx time:',t)

if yolo_pth_model:
    model = torch.load('./weights/wangyi.pt')
    real_time_img = cv2.imread(test_path2)
    real_time_img = cv2.cvtColor(real_time_img,cv2.COLOR_BGR2RGB)
    real_time_img = cv2.resize(real_time_img,(640,640))
    
    cv2.imshow('real_time_img',real_time_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    predict_classification(model=model,
                        trans=real_time_trans,
                        img_or_folder_path=real_time_img,
                        weights_path=None,
                        class_yaml_path=None,
                        fmt='jpg',
                        custom_trans_cv=None,
                        if_show_after_cvtrans=True,
                        if_draw_and_show_result=True)


if yolo_onnx_model:
    
    input_name = 'images'
    output_name = 'output0'
    
    real_time_img = cv2.imread(test_path2)
    real_time_img = cv2.cvtColor(real_time_img,cv2.COLOR_BGR2RGB)
    real_time_img = cv2.resize(real_time_img,(640,640))
    class_info = Data.get_file_info_from_yaml(yolo_class_yaml_path)
    
 
    onnx_engine = Onnx_Engine(yolo_onnx_path,True)
    
    inp = normalize_to_nparray([real_time_img],dtype=np.float32)
    inp = np.transpose(inp,(0,3,1,2))
    
    out_list,t = onnx_engine.run(output_nodes_name_list=None,
                    input_nodes_name_to_npvalue={input_name:inp})
    
    #class_dict = Data.get_file_info_from_yaml(class_yaml_path)
    #p,i = trans_logits_in_batch_to_result(out_list[0])
    
    result = out_list[0]
    a = Yolov5_Post_Processor(None)
    [conts_list,pro_list,cls_list],t2 = a.get_output(result)
    draw_big_rec_list(conts_list,real_time_img)
    
    for cont,pro,cls in zip(conts_list,pro_list,cls_list):
        add_text(real_time_img,f'{pro:.2f}',class_info[cls],cont[0])
        
        
    real_time_img = cv2.cvtColor(real_time_img,cv2.COLOR_RGB2BGR)
    cvshow(real_time_img)
    
    print('onnx time:',t)
    print('post process time:',t2)
    
    
    
    
    
    
    
    

    
    