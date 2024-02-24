import sys

sys.path.append('../..')

from autoaim_alpha.autoaim_alpha.utils_network.data import *
from torchvision import datasets
from params import *


train_dataset = datasets.MNIST(dataset_path,train=True,transform = train_trans)
val_dataset = datasets.MNIST(dataset_path,train=False,transform = val_trans)
Data.save_dict_info_to_yaml({v:k for k,v in train_dataset.class_to_idx.items()},class_yaml_path)

