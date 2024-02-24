# This code is used to check and remove the files in the transform folder if they are not in the bin folder.

import os


import shutil
if __name__ == '__main__':
    
    path = '/mnt/d/datasets/autoaim/roi_binary/based'
    move_to_path = '/mnt/d/datasets/autoaim/roi_binary/4d/transform'
    
    path1 = 'bin'
    path2 = 'transform'
    
    
    path1 = os.path.join(path, path1)
    path2 = os.path.join(path, path2)
    path1_list = os.listdir(path1)
    path2_list = os.listdir(path2)
    count = 0
    for i in path2_list:
        
        if i not in path1_list:
            
            #shutil.move(os.path.join(path2, i), move_to_path)
            os.remove(os.path.join(path2, i))
            count +=1
            print('remove', os.path.join(path2, i))
    
    print(f'Remove {count} files')
    print('Done')
    