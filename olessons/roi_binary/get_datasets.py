import sys
sys.path.append('../..')
from torchvision import transforms, datasets
import torchvision.models.mobilenet
from params import *
from autoaim_alpha.autoaim_alpha.utils_network.data import *
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from params import *
from torch.utils.data import random_split
from params import *


dataset = datasets.ImageFolder(dataset_root_path,trans)

Data.save_dict_info_to_yaml({v:k for k,v in dataset.class_to_idx.items()},'/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/olessons/roi_binary/classes.yaml')






