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

img = cv2.imread(my_test_img)


#oso.traverse('out2',fmt='jpg',deal_func=show_train_trans)

@timing(1,True)
def run():   
    predict_classification(model,
                        val_trans,
                        '3.png',
                        trained2_weights_path,
                        class_yaml_path,
                        'png',
                        custom_trans_cv=write_img_trans,
                        if_show=False,
                        if_print=True)
    
    
_,at = run()
print(at)

    
    
    
    
    
    
    

        
