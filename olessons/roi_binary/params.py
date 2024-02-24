
import cv2
from torchvision import transforms
from autoaim_alpha.autoaim_alpha.utils_network.data import *
dataset_root_path ='/mnt/d/datasets/autoaim/roi_binary'





class_yaml_path = './classes.yaml'
yaml_path = './path.yaml'

log_save_local_path = './local_log'
weights_savelocal_path = './local_weights'

val_split_rate = 1/10





trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((32,32)),
    transforms.ToTensor()
])



batchsize = 30
epochs = 10
learning_rate = 0.0001



