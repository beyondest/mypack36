import sys
sys.path.append('../..')
from autoaim_alpha.autoaim_alpha.utils_network.actions import *
from autoaim_alpha.autoaim_alpha.utils_network.mymodel import *
from autoaim_alpha.autoaim_alpha.utils_network.data import *
from torchvision import datasets,transforms

from torch.utils.data import DataLoader,Dataset
from params import *

train_dataset = datasets.MNIST('./data',train=True,transform = train_trans)
val_dataset = datasets.MNIST('./data',train=False,transform = val_trans)
Data.save_dict_info_to_yaml({v:k for k,v in train_dataset.class_to_idx.items()},class_yaml_path)


train_dataloader = DataLoader(train_dataset,
                              batch_size,
                              shuffle=True)


val_dataloader = DataLoader(val_dataset,
                            batch_size,
                            False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


model = lenet5()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


train_classification(
    model,
    train_dataloader,
    val_dataloader,
    device,
    epochs,
    criterion,
    optimizer,
    weights_save_path,
    show_epoch_interval=1    
)


