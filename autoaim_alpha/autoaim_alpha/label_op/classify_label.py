import os
import sys
sys.path.append('..')
from utils_network.data import *

def make_files_list(root_dir_of_all_kinds_img:str,lable_cor:dict)->list:
    """generate label_list with each : [file_name, label]

    Args:
        root_dir_of_all_kinds_img (str): root/dog,cat,panda /img1,img2,...

    Raises:
        TypeError: _description_

    Returns:
        list: label_list
    """
    
    files_list = []
    
    for parent, dirnames, filenames in os.walk(root_dir_of_all_kinds_img):
        for filename in filenames:
           
            curr_file = parent.split(os.sep)[-1]
            if curr_file in lable_cor:
                labels = lable_cor[curr_file]
            else:
                raise TypeError(f"there are some labels you do not specify,{curr_file}.\nPlease Change params.py")
            files_list.append([os.path.join(curr_file, filename), labels])
    return files_list

def write_txt(label_list:list, save_path:str,name:str ="label.txt", mode:str='w'):
    """save labels to txtfile

    Args:
        content ( list ): list of [file_name:str,label:int]
        filename ( str ): txt_save_path
        mode (str, optional): mode of open. Defaults to 'w'.
    """
    judge=os.path.splitext(save_path)[-1]
    if  judge== ".txt" or os.path.splitext(save_path)[-1] == ".csv":
        final_path= save_path
    elif judge == "":
        final_path = os.path.join(save_path,name)
    else:
        raise TypeError("save_path end wrong ")
    with open(final_path, mode) as f:
        for line in label_list:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # use space as split sign
                    str_line = str_line + str(data) + " "
                else:
                    # each data in line end with \n
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)
if __name__ == "__main__":
    
    
    # you have to change in params.yaml
    datainfo = Data.get_file_info_from_yaml('./params.yaml')
    train_path=datainfo["train_path"]
    txt_path=datainfo["save_txt_path"]
    a=make_files_list(train_path,lable_cor=datainfo["label_corresponding"])
    write_txt(a,txt_path,name="lable.txt",mode="w")
