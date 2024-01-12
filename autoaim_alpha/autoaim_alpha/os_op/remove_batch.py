import os
from os_op.os_operation import *
remove_path='D:/tmdvs/pywork/vsRM/for_calibrate'

def remove_each(goal_path:str,thresh:int=5000):
    list_path=os.listdir(goal_path)
    for i in list_path:
        if os.path.exists(os.path.join(goal_path,i)):
            name=get_name(i)
            #num=int(name)            
            #if num<thresh:
            os.remove(os.path.join(goal_path,i))

        

def remove_kind(work_root_path:str,goal_pack_name:str):
    '''root->batch->pack'''
    batch_list=os.listdir(work_root_path)
    for i in batch_list:
        batch_path=os.path.join(work_root_path,i)
        pack_list=os.listdir(batch_path)
        for j in pack_list:
            if j==goal_pack_name:
                goal_path=os.path.join(batch_path,j)
                remove_each(goal_path)
    else:
        print('removed done')

#remove_kind(work_root_path,'trans')
remove_each(remove_path)