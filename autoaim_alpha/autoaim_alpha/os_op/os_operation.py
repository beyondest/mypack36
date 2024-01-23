import os
from threading import Thread
from time import sleep,ctime
import time

from datetime import datetime
import gzip
from tqdm import tqdm
from typing import Optional,Union


def regular_name(root_path:str,
                 out_path:str,
                 start:int=0,
                 step:int=1,
                 pre_name:str='',
                 suffix_name:str=''):
    '''keep the original form, but change the name into numbers from start to step'''
    path_list=os.listdir(root_path)
    for i in path_list:
        fmt=i.split('.')[-1]
        abs_path=os.path.join(root_path,i)
        out=os.path.join(out_path,pre_name+str(start)+suffix_name+'.'+fmt)
        os.rename(abs_path,out)
        start+=step
        
        
        
def get_name(path:str='forget.txt')->str:
    a=os.path.basename(path)
    b=os.path.splitext(path)[1]
    return a[:len(a)-len(b)]



def copy_function(src,target):
    if os.path.isdir(src) and os.path.isdir(target):
        filelist=os.listdir(src)
        for file in filelist:
            path=os.path.join(src,file)
            if os.path.isdir(path):   
                target1=os.path.join(target,file)
                os.mkdir(target1) 
                copy_function(path,target1)
            else:
                with open(path, 'rb') as rstream:
                    container = rstream.read()
                    path1 = os.path.join(target, file)
                    with open(path1, 'wb') as wstream:
                        wstream.write(container)
        else:
            print('copy succeed')
            
            
            
def traverse(root_path:str,out_path:Optional[str] = None,fmt:Optional[str] = 'jpg',deal_func = None,*args, **kwargs):
    """Traverse each file in root_path in fmt by deal_func
    Notice: deal_func must have abs_path,(out_path) in the first position of all params\n
    Args:
        root_path (str): folder with no extension
        out_path (str): folder with no extension
        fmt (str): _description_
        deal_func (_type_): deal_func(abs_path,outabs_path,*args,**kwargs) 
    """
    lst_root=os.listdir(root_path)
    for i in lst_root:
        abs_path=os.path.join(root_path,i)
        if fmt == None:
            if out_path is not None:
                outabs_path = os.path.join(out_path,i)
                deal_func(abs_path,outabs_path,*args, **kwargs)
            else:
                outabs_path = None
                deal_func(abs_path,*args, **kwargs)
            

        else:
            if os.path.splitext(abs_path)[1]=='.'+fmt:
                if out_path is not None:
                    outabs_path = os.path.join(out_path,i)
                    deal_func(abs_path,outabs_path,*args, **kwargs)
                else:
                    outabs_path = None
                    deal_func(abs_path,*args, **kwargs)
                

def make_packs(root_path:str,
               pre_name:str,
               start_num:int,
               end_num:int,
               suffix_name:str='',
               step:int=1):
    for i in range(start_num,end_num+1):
        
        os.mkdir(os.path.join(root_path,pre_name+str(i)+suffix_name))


def work(work_root_path:str,
         out_path:str,
         work_pack_list:list,
         root_name:str,
         out_name:str,
         deal_fmt:str,
         deal_func):
    '''deal work_pack_list all by deal func, every pack has sub_pack,
    dealfunc truly deal with sub_pack which has same root_name,
    such as :work->1/2/3~ , 1->frame/bin, frame->bin; 2->frame/bin, frame->bin;
    work_root_path=work, out_path=work,pack_list=[1,2],root_name=frame,out_name=bin'''
    for i in work_pack_list:
        root_path=os.path.join(work_root_path,i,root_name)
        out=os.path.join(out_path,i,out_name)
        traverse(root_path,out,deal_fmt,deal_func)
    
def multi_work(work_root_path:str,
               out_path:str,
               root_name:str,
               out_name:str,
               deal_fmt:str,
               deal_func,
               threads:int=3):
    '''deal work_pack_list in work_root_path all by deal func, every pack has sub_pack,
    dealfunc truly deal with sub_pack which has same root_name,
    such as : work->1/2/3~ ,1->frame/bin, frame->bin; 2->frame/bin, frame->bin;
    work_root_path=work,out_path=work, pack_list=[1,2],root_name=frame,out_name=bin'''
    pack_list=os.listdir(work_root_path)
    general_work=[]
    interval=len(pack_list)//threads+1
    for i in range(threads):
        if (i+1)*interval>len(pack_list):
            general_work.append(pack_list[i*interval:len(pack_list)])
        else:
            general_work.append(pack_list[i*interval:(i+1)*interval])
    t=[]
    for i in range(threads):
        t.append(Thread(target=work,args=(work_root_path,out_path,general_work[i],root_name,out_name,deal_fmt,deal_func)))
    for i in range(threads):
        t[i].start()
        print(f'task {i} start at {ctime()}')




def get_current_datetime(if_show:bool = False):
    """get current date and time, return in list

    Returns:
        date,hour,minute,second
    """
    now = datetime.now()
    current_date = now.date()
    current_time = now.time()
    hour, minute, second = current_time.hour, current_time.minute, current_time.second
    if if_show:
        print(f'Time:{current_date}/{hour}:{minute}.{second}')
    
    
    
    return current_date, hour, minute, second

def split_list(input_list, num_segments)->list:
    """Split list into num_segments part

    Args:
        input_list (_type_): _description_
        num_segments (_type_): _description_

    Returns:
        list:list of new segments
    """
    segment_length = len(input_list) // num_segments
    
    remainder = len(input_list) % num_segments
    
    result = []
    
    start = 0
    for i in range(num_segments):
        # one num of remainder, one segment is longer 1 than normal segment
        length = segment_length + 1 if i < remainder else segment_length
        
        result.append(input_list[start:start + length])
        
        start += length
    
    return result

def merge_files(file_path_list:list, merged_file_path:str ,using_gzip:bool = True):
    
    if using_gzip:
        with gzip.open(merged_file_path,'wb') as merged_file:
            for file_path in file_path_list:
                with gzip.open(file_path,'rb') as each_file:
                    merged_file.write(each_file.read())
    else:
        with open(merged_file_path,'wb') as merged_file:
            for file_path in file_path_list:
                with open(file_path,'rb') as each_file:
                    merged_file.write(each_file.read())
    
    
def delete_file(file_path:str):
    try:
        os.remove(file_path)
        print(f'remove:{file_path}')
    except FileNotFoundError:
        print(f'file {file_path} do not exist')
    except:
        print(f'something wrong when remove file {file_path}')
        

def add_suffix_to_end(path:str,suffix:str)->str:
    base,ext = os.path.splitext(path)
    
    return f"{base}{suffix}{ext}"

def change_ext_only(path:str,ext_new:str = '.ttt')->str:
    base,ext = os.path.splitext(path)
    return f'{base}{ext_new}'


