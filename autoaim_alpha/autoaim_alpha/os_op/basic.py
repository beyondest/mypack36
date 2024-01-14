
import sys
sys.path.append('..')
from typing import Any, Union,Optional
from .global_logger import *
from ..utils_network import data

import cv2



def CLAMP(x,scope:list,if_should_be_in_scope:bool = False):
    """Both use for check value in scope or set value in scope

    Args:
        x (_type_): _description_
        scope (list): _description_
        if_should_be_in_scope (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if scope[0] > scope[1]:
        lr1.error(f"OS_OP : scope {scope} wrong , should be [0] < [1] ")
    else:
        if x <scope[0]:
            if if_should_be_in_scope:
                lr1.error(f"OS_OP : value {x} out of scope {scope}" )
            return scope[0]
        elif x > scope[1]:
            if if_should_be_in_scope:
                lr1.error(f"OS_OP : value {x} out of scope {scope}" )
            return scope[1]
        else:
            return x
            
def RCLAMP(x,scope:list,if_should_be_in_scope:bool = False):
    
    if scope[0] > scope[1]:
        lr1.error(f"OS_OP : scope {scope} wrong , should be [0] < [1] ")
    else:
        if x <scope[0]:
            if if_should_be_in_scope:
                lr1.error(f"OS_OP : value {x} out of scope {scope}" )
            return scope[0]
        elif x > scope[1]:
            if if_should_be_in_scope:
                lr1.error(f"OS_OP : value {x} out of scope {scope}" )
            return scope[1]
        else:
            return x

class SV:
    def __init__(self, initial_value, scope,if_strict_in_scope:bool = True):
        self._value = initial_value
        self._if_strict_in_scope = if_strict_in_scope
        if scope[0] > scope[1]:
            scope[1],scope[0] = scope[0], scope[1]
            lr1.error(f"OS_OP : scoped value init failed, scope[0] > scope [1] , auto change to {scope}")
            
        self._scope = scope

    @property
    def value(self):
        if self._value < self._scope[0] or self._value > self._scope[1]:
            lr1.error(f"OS_OP : scoped value out of range, {self._value} not in range: {self._scope}")
            if self._if_strict_in_scope:
                return CLAMP(self._value)
            else:
                return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        
        
class Custome_Context:
    def __init__(self,
                 context_name:str,
                 obj_with_start_end_errorhandler,
                 ) -> None:
        self.context_name = context_name
        self.obj_with_start_end_errorhandler = obj_with_start_end_errorhandler
        if  self.obj_with_start_end_errorhandler._start is None\
            or self.obj_with_start_end_errorhandler._end is None\
            or self.obj_with_start_end_errorhandler._errorhandler is None:
            lr1.critical(f"OS_OP : Context init failed, obj of {context_name} has no _start or _end or _errorhandler") 
            raise TypeError("OP_OP")   
        
    def __enter__(self):
        try:
            self.obj_with_start_end_errorhandler._start()
            lr1.info(f"OS_OP : Enter context {self.context_name} Success")
        except:
            lr1.error(f"OS_OP : Enter Context {self.context_name} failed")
        
        
        
    def __exit__(self,exc_type,exc_value,traceback):
        if exc_type is not None:
            self.obj_with_start_end_errorhandler._errorhandler(exc_value)
        
        self.obj_with_start_end_errorhandler._end()
        lr1.info(f"OS_OP : Exit context {self.context_name} Success")
    

class Custom_Context_Obj:
    def __init__(self) -> None:
        pass
    
    def _start(self):
        raise TypeError("_start method should be override")
    
    def _end(self):
        raise TypeError("_end method should be override")

    def _errorhandler(self):
        raise TypeError("_errorhandler method should be override")
        
        

class Params:
    def __init__(self) -> None:
        pass
    
                
    def __len__(self):
        return len(vars(self))


    
    def print_show_all(self):
        for key, value in vars(self).items():
            print(f"{key} : {value}")
    
    def load_params_from_yaml(self,yaml_path:str):
        
        reflect_dict =vars(self)
        setted_list = []
        info = data.Data.get_file_info_from_yaml(yaml_path)
        
        if len(info) != len(reflect_dict) :
            lr1.error(f"OS_OP : {yaml_path} has wrong params length {len(info)} , expected {len(reflect_dict)}")
            
        for i,item in enumerate(info.items()):
            key,value = item
            if key not in reflect_dict:
                lr1.error(f"OS_OP : params {key} : {value} from {yaml_path} failed, no such key")
            elif key in setted_list:
                lr1.error(f"OS_OP : params {key} dulplicated")
            else:
                reflect_dict[key] = value
                
            setted_list.append(key)  
            
            
    def save_params_to_yaml(self,yaml_path:str,mode = 'w'):
        
        reflect_dict = vars(self)
        data.Data.save_dict_info_to_yaml(reflect_dict,yaml_path,open_mode=mode)
        

        
        
        
def INRANGE(x,scope:list)->bool :
    if scope[0] > scope[1]:
        lr1.error(f'OS_OP : scope {scope} [0] should < [1]')
        scope[1],scope[0] = scope[0], scope[1]
    if scope[0] <= x <= scope[1]:
        return True
    else :
        return False

def CAL_EUCLIDEAN_DISTANCE(pt1,pt2)->float:
    
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5



def CHECK_INPUT_VALID(input,*available):
    
    if input in available:
        return None
    else :
        lr1.error(f"OS_OP : input is not availble, get {input}, expect {available}")
