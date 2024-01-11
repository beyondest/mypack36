import threading
import time
from typing import Union,Optional
class task:
    
    
    def __init__(self,
                 interval_s:Union[float,None],
                 for_circle_func,
                 for_circle_func_deinit=None,
                 params_for_circle_func:Optional[list] = None,
                 params_for_circle_func_deinit:Optional[list] = None,
                 daemon:bool = False
                 ) -> None:
        
        
        self.exit_flag = threading.Event()
        self.interval_s = interval_s
        self.for_circle_func= for_circle_func
        self.for_circle_func_deinit = for_circle_func_deinit
        self.params_for_circle_func = params_for_circle_func
        self.params_for_circle_func_deinit = params_for_circle_func_deinit
        
        
        self.thread = threading.Thread(target=self.main,daemon=daemon)
        self.ifend =False
        
    def main(self):
        while not self.exit_flag.is_set():
            
            self.for_circle_func(*self.params_for_circle_func)
            time.sleep(self.interval_s)
            
           
        
        
    def start(self,if_clear:bool = True):
        
        if if_clear:
            self.exit_flag.clear()
        self.thread.start()
    
    def end(self):
        """If you want to end a task, must use THIS FUNCTION
        """
        self.exit_flag.set()
        self.thread.join()
        
        if self.for_circle_func_deinit is not None:
            self.deinit()
        self.ifend = True
                
    def deinit(self):
        self.for_circle_func_deinit(*self.params_for_circle_func_deinit)

    
    
    def stop(self,stop_time_s:float=-1):
        """Warning: this will occupy MAIN THREAD of py to stop

        Args:
            stop_time_s (float): _description_
        """
        if stop_time_s == -1:
            
            self.exit_flag.set()
        else:
            
            self.exit_flag.set()
            time.sleep(stop_time_s)
            self.exit_flag.clear()
        
        
    def restart(self):
        self.exit_flag.clear()
        
        
def keyboard_control_task(tasks_list:list,if_all_control_by_ctrlC:bool = True, main_func:None=None,main_func_params:Optional[list] = None):
    """enter 2 to end task 2 in tasks_list
       WARNING: MAIN THREAD of py will stop here until all end
    Args:
        tasks_list (list): [task0,task1,task2...]
    """
    if not isinstance(tasks_list[0],task):
        raise TypeError('Wrong tasks_list component')
    
    task_nums = len(tasks_list)
    keyboard_list = [i for i in range(task_nums)]
    cor_dict = {k:v for k,v in zip(keyboard_list,tasks_list)}
    count =0
    
    
    try:
        
        for i in keyboard_list:
            cor_dict[i].start()

        while True:
            if not if_all_control_by_ctrlC:
                
                key = int(input("ENTER nums to end task\n"))
                if key in cor_dict:
                    cor_dict[key].end()
                    cor_dict.pop(key)
                    count+=1
                if count ==task_nums:
                    break
            if main_func is not None:
                if main_func_params is not None:
                    
                    main_func(*main_func_params)
                else:
                    main_func()
            
            
    except:
        for i in keyboard_list:
            print(i)
            
            cor_dict[i].end()
    
        
    finally:
        print("ALL DONE")
        pass
    

class test:
    def __init__(self) -> None:
        
        self.a = 1
    

if __name__ == "__main__":
    
    
    data0 = test()
    data = {'a':1}
    def f1(for_op):
        print('f1 only show',for_op.a)
        
    def deinitf1(for_op):
        print('deinit1',for_op.a)
    
    def f2(for_op):
        for_op.a+=1
        print(f'f2 change data:data++ {for_op.a}')
        
    def deinitf2(for_op):
        print('deinit2',for_op.a)
    
    task_a = task(0.3,
                  for_circle_func=f1,
                  for_circle_func_deinit=deinitf1,
                  params_for_circle_func=[data0],
                  params_for_circle_func_deinit=[data0],
                  )   
    task_b = task(0.9,
                  for_circle_func=f2,
                  params_for_circle_func=[data0],
                  for_circle_func_deinit=deinitf2,
                  params_for_circle_func_deinit=[data0])
    
    keyboard_control_task([task_a,task_b])
    
    
    
    
    
    
    
    
    
    
    
    