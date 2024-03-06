

from typing import Any, Union,Optional
from .global_logger import *
from ..utils_network import data
from .decorator import *
import numpy as np


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
                 ignore_error_type:list = None
                 ) -> None:
        self.context_name = context_name
        self.obj_with_start_end_errorhandler = obj_with_start_end_errorhandler
        if  self.obj_with_start_end_errorhandler._start is None\
            or self.obj_with_start_end_errorhandler._end is None\
            or self.obj_with_start_end_errorhandler._errorhandler is None:
            lr1.critical(f"OS_OP : Context init failed, obj of {context_name} has no _start or _end or _errorhandler") 
            raise TypeError("OP_OP")   
        self.ignore_error_type = ignore_error_type if ignore_error_type is not None else []
        
    def __enter__(self):
        try:
            self.obj_with_start_end_errorhandler._start()
            lr1.info(f"OS_OP : Enter context {self.context_name} Success")
        except:
            lr1.error(f"OS_OP : Enter Context {self.context_name} failed")
        
        
        
    def __exit__(self,exc_type,exc_value,traceback):
        if exc_type is not None and exc_type not in self.ignore_error_type:
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


def TRANS_RVEC_TO_ROT_MATRIX(rvec:np.ndarray)->np.ndarray:

    if rvec is None:
        lr1.error(f"OS_OP : TRANS_RVEC_TO_ROT_MATRIX failed, rvec is None")
        return None
    
    theta = np.linalg.norm(rvec)
    if theta == 0:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return rot_matrix



class Score_Obj:
    def __init__(self,
                 name:str,
                 attributes_list:list,
                 score_accumulation_method_list:list,
                 weights_list:list,
                 reference_list:Optional[list]=None):
        """

        Args:
            name (str): _description_
            attributes_list (list): _description_
            score_accumulation_method_list (list): _description_
                if 'add' : add score to current score
                if'mul' : multiply score to current score
                if 'pow' : power score to current score

        Raises:
            TypeError: _description_
        """
        
        if len(score_accumulation_method_list) != len(attributes_list):
            lr1.critical(f"OS_OP : score_accumulation_method_list length {len(score_accumulation_method_list)} not equal to attributes_list length {len(attributes_list)}")
            raise TypeError(f"score_accumulation_method_list length {len(score_accumulation_method_list)} not equal to attributes_list length {len(attributes_list)}")
        
        for i in score_accumulation_method_list:
            CHECK_INPUT_VALID(i,'add','mul','pow')
        
        self.name = name
        self.score_list = []
        self.score_accumulation_method_list = score_accumulation_method_list
        self.attributes_list = attributes_list
        self.reference_list = reference_list
        
        for id,attribute in enumerate(attributes_list):
            
            score = self.turn_attribute_to_score(id,attribute,reference_list[id] if reference_list is not None else None)
            self.score_list.append(score * weights_list[id])
            
    def turn_attribute_to_score(self,
                                 attribute_id,
                                 attribute_value,
                                 reference_value:Optional[float]=None)->float:
        
        raise NotImplementedError("OS_OP : turn_attribute_to_score method should be override")
    
    def show_each_attribute_score(self):
        
        lr1.info(f"Score_Obj : {self.name} : each attribute score :")
        for i,score in enumerate(self.score_list):
            lr1.info(f"Score_Obj : attribute {i} : {self.attributes_list[i]} reference {self.reference_list[i]} score {score}")

class Score_Keeper:
    
    def __init__(self) -> None:
        self.score_obj_name_to_score = {}
        
    def add_score_obj(self,score_obj:Score_Obj):
        
        score = 0
        for i,score_method in enumerate(score_obj.score_accumulation_method_list):
            
            if score_method == 'add':
                score += score_obj.score_list[i]
            elif score_method =='mul':
                score *= score_obj.score_list[i]
            elif score_method == 'pow':
                score **= score_obj.score_list[i]
            
        if score_obj.name in self.score_obj_name_to_score:
            lr1.critical(f"OS_OP : score_obj {score_obj.name} already in score_keeper")
            raise TypeError(f" score_obj {score_obj.name} already in score_keeper")
            
        self.score_obj_name_to_score[score_obj.name] = score
    
    def get_name_and_score_of_score_by_rank(self,rank:int)->list:
        """Get name of score by rank, rank start from 0; Remember update_rank() before use this method

        Args:
            rank (int): _description_

        Raises:
            TypeError: _description_

        Returns:
            name,score: _description_
        """
        if rank < 0 or rank >= len(self.score_obj_name_to_score):
            lr1.critical(f"OS_OP : rank {rank} out of range, should be 0 <= rank < {len(self.score_obj_name_to_score)}")
            raise TypeError(f"rank {rank} out of range, should be 0 <= rank < {len(self.score_obj_name_to_score)}")

        
        return list(self.score_obj_name_to_score.keys())[rank],list(self.score_obj_name_to_score.values())[rank]
    
    
    def update_rank(self):
        
        self.score_obj_name_to_score = dict(sorted(self.score_obj_name_to_score.items(), key=lambda item: item[1],reverse=True))


def CAL_COS_THETA(v1,v2)->float:
    """Calculate cos theta between two vectors

    Args:
        v1 (np.ndarray): _description_
        v2 (np.ndarray): _description_

    Returns:
        float: _description_
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0
    else:
        return np.dot(v1,v2) / (v1_norm * v2_norm)


def SHIFT_LIST_AND_ASSIG_VALUE(lst:list,value)->list:
    """Shift list to right and assign value to the lst[0]

    Args:
        lst (list): _description_
        value (_type_): _description_

    Returns:
        list: _description_
    """
    lst.insert(0,value)
    lst.pop()
    return lst

@timing(1)
def BISECTION_METHOD(f,a,b,e:float = 1e-6)->float:
    
    while abs(b-a) > e:
        c = (a+b)/2
        if f(c) == 0:
            return c
        elif f(a)*f(c) < 0:
            b = c
        else:
            a = c
            
    return (a+b)/2




def TRANS_UNIX_TIME_TO_T(unix_time:float,
                         zero_unix_time:float)->tuple:
    """

    Args:
        unix_time (float): _description_
        zero_unix_time (float): _description_

    Returns:
        minute (int): _description_
        second (int): _description_
        second_frac (float): _description_
    """
    dt = unix_time - zero_unix_time
    
    minute = int(dt//60)
    second = int(dt%60)
    second_frac = dt%1
    
    return minute, second, second_frac

def TRANS_T_TO_UNIX_TIME(minute:int,
                         second:int,
                         second_frac:float,
                         zero_unix_time:float)->float:
    """

    Args:
        minute (int): _description_
        second (int): _description_
        second_frac (float): _description_
        zero_unix_time (float): _description_

    Returns:
        float: _description_
    """
    return minute*60 + second + second_frac + zero_unix_time