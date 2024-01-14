import numpy as np
import torch
import sys
sys.path.append('../..')
from utils_network.data import *
from utils_network.mymodel import *
import cv2
import torch
from torchvision import transforms
from PIL import Image
from olessons.multiclassification.params import *
from img.img_operation import *
from utils_network.actions import *
from os_op.decorator import *
test_mnist_img = 'out2/mnist_0.jpg'
my_test_img = 'test.png'

model = lenet5() 

ori_img = cv2.imread('3.png')


model.load_state_dict(torch.load(trained2_weights_path,map_location=device))

model.eval()
idx_to_class = Data.get_file_info_from_yaml(class_yaml_path)


#
img = write_img_trans(ori_img)
input_tensor = val_trans(img)
input_tensor = torch.unsqueeze(input_tensor,0)
#



@timing(circle_times=10000,if_show_total=True)
def run():

    '''img2 = cv2.imread('3.png')
    img2 = write_img_trans(img2)
    input_tensor2 = val_trans(img2)
    input_tensor2 = torch.unsqueeze(input_tensor2,0)


    input_all = torch.cat((input_tensor,input_tensor2),dim=0)
'''
    with torch.no_grad():
        results = model(input_tensor)
    return results

result,elapesd_time = run()
print(f'avg_spending: {elapesd_time:2f}')

logits = result

softmax_result = torch.softmax(logits,dim=1)
max_result = torch.max(softmax_result,dim=1)
max_index = max_result.indices
max_probability = max_result.values
print(max_probability)
print(max_index)



