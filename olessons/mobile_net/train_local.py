
from torchvision import transforms, datasets
import torchvision.models.mobilenet
import sys
sys.path.append('../../')
from params import *
from utils_network.data import *
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
import os_op.os_operation as oso
from utils_network.actions import *
from utils_network.mymodel import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device:{device}')



val_dataset = datasets.ImageFolder(val_root_path,val_trans)
train_dataset = datasets.ImageFolder(train_root_path,train_trans)


val_dataloader = DataLoader(val_dataset,
                        batchsize,
                        shuffle=False,
                        num_workers=1)

train_dataloader = DataLoader(train_dataset,
                              batchsize,
                              shuffle=True,
                              num_workers=2)


model = mobilenet_v2(num_classes=2)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


train_classification(model,
                     train_dataloader,
                     val_dataloader,
                     device,
                     epochs,
                     criterion,
                     optimizer,
                     weights_savelocal_path,
                     save_interval=1,
                     show_step_interval=10,
                     show_epoch_interval=1,
                     log_folder_path=log_save_local_path)



















            
            
    
    
    






