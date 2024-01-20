import numpy as np
import random 
import time
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('..')
from .mymath import *
import torch
from .mymodel import *
from ..os_op import os_operation as oso
import os
import yaml
from torch.utils.data import Dataset,DataLoader
import gzip
import pickle
from torch.utils.data import TensorDataset,Subset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import threading
import cv2
from PIL import Image
import PIL as pil
from typing import Union,Optional
import onnxruntime
from ..img.tools import normalize

def tp(*args):
    for i in args:
        print(type(i))







class Data:
    def __init__(self,file_path:Union[str,None]=None,seed:int=10) -> None:
        random.seed(seed)        
        np.random.seed(seed)
    

  
    
    @classmethod
    def show_nplike_info(cls,
                     for_show:Union[np.ndarray,list,tuple],
                     precision:int=2,
                     threshold:int=1000,
                     edgeitems:int=3,
                     linewidth:int=75,
                     suppress:bool=False):
        '''
        precision: how many bit after point for float nums\n
        threshold: how many nums to show a summary instead of show whole array\n
        edgeitems: how many nums to show in a summary at begin of a row\n
        linewidth: how many nums to show in a row\n
        suppress: if True , always show float nums NOT in scientifc notation
        '''
        np.set_printoptions(precision=precision,
                            threshold=threshold,
                            edgeitems=edgeitems,
                            linewidth=linewidth,
                            suppress=suppress)
        torch.set_printoptions(precision=precision,
                               threshold=threshold,
                               edgeitems=edgeitems,
                               linewidth=linewidth)
        
        if isinstance(for_show,np.ndarray): 
            print('shape:',for_show.shape)
            print('dtype:',for_show.dtype)
            print('max:',for_show.max())
            print('min:',for_show.min())
            print(for_show)
        elif isinstance(for_show,list):
            count=0
            for i in for_show:
                if not isinstance(i,torch.Tensor) and not isinstance(i,np.ndarray) and not isinstance(i,int):
                    raise TypeError(f'unsupported input type {type(i)},only support np.ndarray and torch.Tensor')
                if isinstance(i,int):
                    print(f'*********{count}*********')
                    print('type:int,value:',i)
                else:
                    
                    print(f'*********{count}*********')
                    print('shape:',i.shape)
                    print('dtype:',i.dtype)
                    print('max:',i.max())
                    print('min:',i.min())
                    print('content:',i)
                count+=1
            
                
                

    def make_taylor_basis(self,column:np.ndarray,order:int=3)->np.ndarray:
        '''
        use column vector x to generate [1 x x^2 x^3 ... x^order] matrix\n
        return matrix  
        
        '''
        column=np.copy(column)
        out=np.ones_like(column)
        for power_index in range(1,order+1):
            out=np.c_[out,np.power(column,power_index)]   
        return out

    def make_fourier_basis(self,column:np.ndarray,order:int=100):
        '''
        use column vector x to generate [1 cosx sinx cos2x sin2x ... cos(order*x) sin(order*x)] matrix\n
        return matrix  
        
        '''
        column=np.copy(column)
        out=np.ones_like(column)    #simulate cos0x
        for multiple_index in range(1,order):
            out=np.c_[out,np.cos(multiple_index*column)]
            out=np.c_[out,np.sin(multiple_index*column)]
        return out
  
    @classmethod
    def save_feature_sample_nums_to_yaml(cls,feature_nums:int,sample_nums:int,save_path:str="./train_param.yaml",open_mode:str="w"):
        """save feature nums and sample nums to yaml file after getting dataset

        Args:
            feature_nums (int): _description_
            sample_nums (int): _description_
            save_path (str, optional): _description_. Defaults to "./train_param.yaml".
            open_mode (str, optional): _description_. Defaults to "w".
        """
        forsave={
            "feature_nums": feature_nums,
            "sample_nums":  sample_nums,
            "saving_time": time.asctime()
        }      
        with open(save_path, open_mode) as file:
            yaml.dump(forsave,file)
            print(f"feature_nums and sample_nums saved to {save_path}")
   
    @classmethod
    def get_feature_sample_nums_from_yaml(cls,
                                          yaml_path:str,
                                          open_mode:str="r")->tuple:
        """get feature nums and sample nums from yaml file 

        Args:
            yaml_path (str): _description_
            open_mode (str, optional): _description_. Defaults to "r".

        Returns:
            tuple: _description_
        """
        with open(yaml_path,open_mode) as file:
            print(yaml_path)
            print(os.getcwd())
            data = yaml.safe_load(file)

            
            
            return data["feature_nums"],data["sample_nums"]
    
    
    @classmethod 
    def get_file_info_from_yaml(cls,yaml_path:str,open_mode:str = 'r')->dict:
        
        with open(yaml_path,open_mode) as file:
            config = yaml.safe_load(file)
            
        return config
    
    
    @classmethod
    def save_dict_info_to_yaml(cls,dict_info:dict,yaml_path:str,open_mode:str = 'w'):
        
        with open(yaml_path,open_mode) as file:
            yaml.dump(dict_info,file,default_flow_style=False)
        print(f'Dict_info saved to yamlfile in {yaml_path}')
    
    @classmethod
    def save_dataset_to_pkl(cls,
                            dataset:Dataset,
                            pkl_path:str,
                            if_multi:bool = False,
                            max_workers:int = 5):
        """Use 'ab' to write pkl.gz file, first data is length of dataset

        Args:
            dataset (Dataset): _description_
            pkl_path (str): _description_
            if_multi (bool, optional): _description_. Defaults to True.
        """
        print(f'ALL {len(dataset)} samples to save')

        global_lock = threading.Lock()
        proc_bar = tqdm(len(dataset),desc="Saving to pkl:")
        
        with gzip.open(pkl_path,'ab') as f:
            pickle.dump(len(dataset),f)
            
        def save_sample(sample, file_path):
            while global_lock.locked():
                time.sleep(0.02)
                continue
            global_lock.acquire()
            
            x, y = sample
            
            proc_bar.update(1)
            with gzip.open(file_path, 'ab') as f:
                pickle.dump((np.asanyarray(x),np.asanyarray(y)), f)
            global_lock.release() 
                
        if if_multi:
               
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  
                futures = [executor.submit(save_sample, sample, pkl_path) for sample in dataset]

                concurrent.futures.wait(futures)
        else:
            for sample in dataset:
                save_sample(sample,pkl_path)
        
        proc_bar.close()
        print(f'dataset saved to {pkl_path}')
        print('Notice: when you need to open it, please include your dataset defination in open code')

    
    @classmethod
    def save_dataset_to_pkl2(cls,
                             dataset,
                             pkl_path,
                             max_workers):
        """Dont use this now

        Args:
            dataset (_type_): _description_
            pkl_path (_type_): _description_
            max_workers (_type_): _description_
        """
        length = len(dataset)
        print(f'ALL {length} samples to save')
        all_sample_list = [i for i in dataset]
        segment_list = oso.split_list(all_sample_list,max_workers)
        temp_file_path_list= [f'temp{i}.pkl.gz' for i in range(max_workers)]
        
        params_list = list(zip(segment_list,temp_file_path_list))
        proc_bar = tqdm(length,desc="Saving to pkl:")
        
        def save_sample(part_sample_list, file_path):
            for x,y in part_sample_list:
                proc_bar.update(1)
                with gzip.open(file_path, 'ab') as f:
                    
                    pickle.dump((np.asanyarray(x),np.asanyarray(y)), f)
        
        
               
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  
            futures = [executor.submit(save_sample, part_sample_list, pkl_path) for part_sample_list,pkl_path in params_list]

            concurrent.futures.wait(futures)

        oso.merge_files(temp_file_path_list,pkl_path)
        for i in temp_file_path_list:
            oso.delete_file(i)
        
        proc_bar.close()
        print(f'dataset saved to {pkl_path}')
        print('Notice: when you need to open it, please include your dataset defination in open code')

    @classmethod
    def save_dataset_to_npz(cls,
                            dataset,
                            npz_path:str
                            ):
        print(f'ALL {len(dataset)} samples to save')
        
        proc_bar = tqdm(len(dataset),"changing dataset type to ndarray:")
        X_list = []
        y_list = []
        for X,y in dataset:
            X_list.append(X)
            y_list.append(y)
            proc_bar.update(1)
        X = np.asanyarray(X_list)
        y = np.asanyarray(y_list)
        np.savez_compressed(npz_path,X = X,y = y)
        proc_bar.close()
        print(f'dataset saved to {npz_path}')
        print('Notice: when you need to open it, please include your dataset defination in open code')
  
       
    @classmethod
    def get_dataset_from_npz(cls,
                             npz_dataset_obj:Dataset,
                             npz_path:str):
        oso.get_current_datetime(True)
        data = np.load(npz_path)
        X = torch.from_numpy(data['X'])
        y = torch.from_numpy(data['y'])
        dataset = TensorDataset(X,y)
        print(f'{npz_path}   npz dataset length is {len(dataset)}')
        oso.get_current_datetime(True)
        return dataset

    
    @classmethod
    def get_dataset_from_pkl(cls,
                             pkl_dataset_obj:Dataset,
                             pkl_save_path:str,
                             open_mode:str = 'rb')->Dataset:
        def load_data(file_path):
            with gzip.open(file_path, open_mode) as f:
                data = []
                length = pickle.load(f)
                proc_bar = tqdm(length,desc=f"Reading from {file_path}:")
                while True:
                    try:
                        sample = pickle.load(f)
                        data.append(sample)
                        proc_bar.update(1)
                    except EOFError:
                        proc_bar.close()
                        print('Finish reading')
                        break
            return data
        loaded_data = load_data(pkl_save_path)
        x = np.asanyarray([sample[0] for sample in loaded_data])
        y = np.asanyarray([sample[1] for sample in loaded_data])
        X = torch.tensor(x)
        y = torch.tensor(y)

        d = TensorDataset(X,y)
        return d
    
    
    @classmethod
    def get_path_info_from_yaml(cls,yaml_path:str)->tuple:
        """get path info from yaml 

        Args:
            yaml_path (str): _description_

        Returns:
            train_path,val_path,weights_save_path,log_save_folder
        """
        info =Data.get_file_info_from_yaml(yaml_path)
        train_path = info['train_path']
        val_path = info['val_path']
        weights_save_path = info['weights_save_path']
        log_save_folder = info['log_save_folder']
        return train_path,val_path,weights_save_path,log_save_folder
    
    
    @classmethod
    def save_model_to_onnx(cls,
                           model:torch.nn.Module,
                           output_abs_path:str,
                           dummy_input:torch.Tensor,
                           trained_weights_path:Union[str,None] = None,
                           input_names = ['input'],
                           output_names = ['output'],
                           if_dynamic_batch_size:bool = True,
                           dynamic_axe_name = 'batch_size',
                           dynamic_axe_input_id = 0,
                           dynamic_axe_output_id = 0,
                           opt_version:int=10
                           ):
        
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if if_dynamic_batch_size == False:
            dynamic_axes = None
        else:
            dynamic_axes = {input_names[0]: {dynamic_axe_input_id: dynamic_axe_name}, 
                            output_names[0]: {dynamic_axe_output_id: dynamic_axe_name}}

        if trained_weights_path is not None:
            model.load_state_dict(torch.load(trained_weights_path,map_location=device))
        
        model.eval()
        # quatized model, but notice not all platforms onnx run will support this, so you need to add ATEN_FALLBACK 
        #q_model = quantize_dynamic(model,dtype=torch.qint8)
        
        torch.onnx.export(model=model,                          #model to trans
                        args=dummy_input,                       #dummy input to infer shape
                        f=output_abs_path,                      #output onnx name 
                        verbose=True,                           #if show verbose information in console
                        export_params=True,                     #if save present weights to onnx
                        input_names=input_names,                #input names list,its length depends on how many input your model have
                        output_names=output_names,              #output names list
                        training=torch.onnx.TrainingMode.EVAL,  #EVAL or TRAINING or Preserve(depends on if you specify model.eval or model.train)
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,   #ONNX or ONNX_FALLTROUGH or ONNX_ATEN_FALLBACK  or ONNX_ATEN, ATEN means array tensor library of Pytorch
                                                                                         # fallback to onnx or fallthrough to aten, use aten as default, aten better for torch, but onnx is more compat
                        opset_version=opt_version,                       #7<thix<17
                        do_constant_folding=True,               #True
                        dynamic_axes = dynamic_axes,            #specify which axe is dynamic
                        keep_initializers_as_inputs=False,      #if True, then you can change weights of onnx if model is same   
                        custom_opsets=None                 #custom operation, such as lambda x:abs(x),of course not so simple, you have to register to pytorch if you want to use custom op
                        )                 

        print(f'ori_onnx model saved to {output_abs_path}')
    

def get_subset(dataset:Dataset,scope:list = [0,500])->Dataset:
    
    return Subset(dataset,[i for i in range(scope[0],scope[1])])  
 
            
class dataset_pkl(Dataset):
    def __init__(self,pkl_path:str) -> None:
        super().__init__()
        
        self.path = pkl_path
        with gzip.open(pkl_path,'rb') as f:
            self.length = pickle.load(f)
            self.position = f.tell()
        
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index)->tuple:
        with gzip.open(self.path,'rb') as f:
            f.seek(self.position)
            X,y = pickle.load(f)
            self.position = f.tell()
        
        return torch.from_numpy(X),torch.from_numpy(y)
   
            


    
    






class PIL_img_transform:
    def __init__(self,
                 cv_custom_transform,
                 total_input_channel_type:str = 'rgb''bgr''gray') -> None:
        """API for custom cv2 transform function

        Args:
            cv_custom_transform: 
            input MUST be BGR image;\n
            output controled by custom_transform MUST fit net input;
            input_channel_type : depends on what PIL will read in
            
        """
        self.cv_custom_transfomr = cv_custom_transform
        self.total_input_channel_type = total_input_channel_type
        
        
    def __call__(self,ori_X):
        if self.cv_custom_transfomr is None:
            raise TypeError('custom trans form cannot be None')
        
        nparray = np.array(ori_X)
        
        if self.total_input_channel_type == 'rgb':
            try:
                cv_img = cv2.cvtColor(nparray,cv2.COLOR_RGB2BGR)
            except:
                raise TypeError('input img is not rgb form, shape is',nparray.shape)
        elif self.total_input_channel_type == 'gray':
            try:
                cv_img = cv2.cvtColor(nparray,cv2.COLOR_GRAY2BGR)
            except:
                raise TypeError('input img is not gray form, shape is',nparray.shape)
            
        elif self.total_input_channel_type == 'bgr':
            if len(nparray.shape) <3:
                raise TypeError('input img not match bgr form, shape is',nparray.shape)
            cv_img = nparray
        
        else:
            raise TypeError(f"Wrong input channel type {self.total_input_channel_type}, support rgb,gray,bgr")
        
        
        
            
        cv_img = self.cv_custom_transfomr(cv_img)
        
        
        
        
        return cv_img
    



    
    
    
        
        
    
    



def normalize_to_nparray(img_or_imglist:Union[np.ndarray,list],
                      dtype:type = np.float32)->Union[np.ndarray,None]:
    if img_or_imglist is None:
        return None
    if isinstance(img_or_imglist,list):
        
        inp = np.asanyarray(img_or_imglist)
        if len(inp.shape) == 3:
            inp = np.expand_dims(inp,axis=1)
            
    else:
        inp = np.expand_dims(img_or_imglist,axis=0)
        inp = np.expand_dims(inp,axis=0)
    
    
    return normalize(inp).astype(dtype=dtype)
    


        
    











if __name__=="__main__":
    path="../pth_folder/test.pth"
    '''def func0(x):
        return np.power(x,2)

    def func1(x):
        return np.power(x,3)

    
    model1=simple_2classification(feature_nums=4,feature_expand=True,correlation=[[[0],[0],func0],[[1],[1],func0],[[0],[2]],[[1],[3]]])
    checkpoints=torch.load(path)
    model1.load_state_dict(checkpoints)
    print(model1.state_dict())
    
    hp_pts=Data.get_hyperplane_pts(model1,pt_nums=100)
    f=Data.plt_figure()
    ax_index=f.plt_point(x=hp_pts[:,0],y=hp_pts[:,1],xlim=(-5,5),ylim=(-5,5))
    plt.show()
    '''
    
    

    

    
    
    
    
    
    
    
    
    