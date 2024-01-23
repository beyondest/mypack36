import os
import argparse

def regular_name(root_path:str,
                 out_path:str,
                 start:int=0,
                 step:int=1,
                 pre_name:str='',
                 suffix_name:str=''):
    '''keep the original form, but change the name into numbers from start to step'''
    path_list=os.listdir(root_path)
    start = start-step
    
    for i in path_list:
        start+=step
        
        fmt=i.split('.')[-1]
        abs_path=os.path.join(root_path,i)
        out=os.path.join(out_path,pre_name+str(start)+suffix_name+'.'+fmt)
        os.rename(abs_path,out)
        
    print(f'Rename {len(path_list)} files successfully!')
        
        
if __name__ == '__main__':
       
    pareser = argparse.ArgumentParser()
    pareser.add_argument('--root_path',type=str,default=os.getcwd(),help='root path')
    pareser.add_argument('--out_path',type=str,default=os.getcwd(),help='output path')
    pareser.add_argument('--start',type=int,default=0,help='start number')
    pareser.add_argument('--step',type=int,default=1,help='step number')
    pareser.add_argument('--pre_name',type=str,default='',help='prefix name')
    pareser.add_argument('--suffix_name',type=str,default='',help='suffix name')
    args = pareser.parse_args()
    
    root_path = args.root_path
    out_path = args.out_path
    start = args.start
    step = args.step
    pre_name = args.pre_name
    suffix_name = args.suffix_name
    
    path = os.path.abspath(root_path)
    
    print('Current working directory is: ', path)
    print('Output directory is: ', out_path)

    regular_name(path,path,start=start,step=step,pre_name=pre_name,suffix_name=suffix_name)
