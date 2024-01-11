
#   Notice : Paths here are just used for saving and reading datasets on your own pc, when in colab, use path.yaml to train
from torchvision import transforms
train_root_path ='/mnt/d/datasets/petimages/train'
train_tfrecords_path = '/mnt/d/datasets/record/pet_train_224.tfrecords'
train_pkl_path = '../../datasets/train.pkl.gz'
train_npz_path = '../../datasets/train.npz'
train_hdf5_path = '../../datasets/train.hdf5'


val_root_path = '/mnt/d/datasets/petimages/val'
val_tfrecords_path = '../../datasets/pet_val_224.tfrecords'
val_pkl_path = '../../datasets/val.pkl.gz'
val_npz_path = '../../datasets/val.npz'
val_hdf5_path = '../../datasets/val.hdf5'





class_yaml_path = './classes.yaml'
yaml_path = './path.yaml'

log_save_local_path = './local_log'
weights_savelocal_path = './local_weights'

predict_cat_path = './res/cat.1.jpg'
predict_dog_path = './res/dog3.jpg'
trained_weights_path = './weights/weights.0.83.9.pth'


ori_onnx_path = './weights/old83.onnx'
opt_qua_onnx_path = './weights/Q83.onnx'




flip_probability = 0.5
mean,std = [0.485,0.456,0.406],[0.229,0.224,0.225]
train_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(flip_probability),
    transforms.Normalize(mean,std)
])
val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean,std)
])



batchsize = 30
epochs = 10
learning_rate = 0.0001



