import sys
sys.path.append('../../')
from params import *
from utils_network.data import *
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader




train_dataset  =  datasets.ImageFolder(train_root_path,train_trans)
val_dataset = datasets.ImageFolder(val_root_path,val_trans)


Data.save_dict_info_to_yaml({v:k for k,v in train_dataset.class_to_idx.items()},class_yaml_path)
#Data.save_dataset_to_hdf5(val_dataset,val_hdf5_path)








