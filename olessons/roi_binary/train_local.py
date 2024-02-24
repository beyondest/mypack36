import sys
sys.path.append('../..')
from torchvision import transforms, datasets
import torchvision.models.mobilenet
from params import *
from autoaim_alpha.autoaim_alpha.utils_network.data import *
from autoaim_alpha.autoaim_alpha.utils_network.actions import *
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from params import *
from torch.utils.data import random_split
from params import *


from params import *
from torch.utils.data import random_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'Using device:{device}')

dataset = datasets.ImageFolder(dataset_root_path,trans)

val_size = int(len(dataset)*val_split_rate)

train_dataset,val_dataset = random_split(dataset,[len(dataset)-val_size,val_size])



val_dataloader = DataLoader(val_dataset,
                        batchsize,
                        shuffle=False,
                        num_workers=1)

train_dataloader = DataLoader(train_dataset,
                              batchsize,
                              shuffle=True,
                              num_workers=2)


model = QNet(11)

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



















            
            
    
    
    






