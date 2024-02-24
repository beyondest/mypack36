import numpy as np
import torch
import sys
sys.path.append('../../utils_network/')
from autoaim_alpha.autoaim_alpha.utils_network.data import *
from autoaim_alpha.autoaim_alpha.utils_network.mymodel import *
from torchvision import transforms,datasets
from autoaim_alpha.autoaim_alpha.img.img_operation import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_path='./test.png'
epochs = 5
learning_rate = 0.001
batch_size = 30


dataset_path = '/home/liyuxuan/vscode/pywork_linux/mypack/olessons/multiclassification/data'
weights_save_path='./weights2'
trained_weights_path = './weights2/weights.0.92.1.pth'
trained2_weights_path = './weights2/best71.4.pth'

class_yaml_path = './class.yaml'


show_trans = transforms.Compose([

    transforms.RandomAffine(degrees=0,translate=(0.2,0.1),scale=(1,2),shear=45)
])

def cv_trans_train(img:np.ndarray):

    img = add_noise_circle(img,circle_radius_ratio=1/6,noise_probability=1/3)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    return img
    
def cv_trans_val(img:np.ndarray):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    return img


def write_img_trans(img:np.ndarray)->np.ndarray:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #_,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    
    img = cv2.bitwise_not(img)
    img = cv2.dilate(img,np.ones((int(img.shape[0]/80),int(img.shape[1]/80))),iterations=3)
    
    img = cv2.resize(img,(28,28))
    _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    return img

def show_train_trans(abs_path):
    img = Image.open(abs_path)
    img = cv_trans_train(img)


# Notice : original mnist dataset img is grayscale!!!

train_trans = transforms.Compose([
    transforms.Resize((40,35)),
    transforms.RandomCrop(28),
    transforms.RandomAffine(degrees=0,translate=(0.2,0.1),scale=(1,1.4),shear=30),
    PIL_img_transform(cv_trans_train,'gray'),
    transforms.ToTensor()
])


val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    PIL_img_transform(cv_trans_val,'gray'),
    transforms.ToTensor(),
    transforms.Resize((28,28))
])

