
import sys
sys.path.append('..')
from typing import Any, Union,Optional
from .global_logger import *


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
    

