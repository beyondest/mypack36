
import sys
sys.path.append('../..')
from utils_network.data import *
from torchvision import transforms

dataset_root_path ='/mnt/d/datasets/roi_binary_old/train'





class_yaml_path = './classes.yaml'
yaml_path = './path.yaml'

log_save_local_path = './local_log'
weights_savelocal_path = './local_weights'


def cv_trans(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    _,img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY)


train_trans = transforms.Compose([
    PIL_img_transform(),    
    transforms.RandomResizedCrop(32)
])

val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32)),
])



batchsize = 30
epochs = 10
learning_rate = 0.0001



