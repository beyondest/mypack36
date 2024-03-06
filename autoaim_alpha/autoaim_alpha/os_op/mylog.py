import logging
from datetime import datetime
from typing import Optional,Union
import os




class NO_Logging_logger:
        def __init__(self) -> None:
            pass
        @classmethod
        def debug(cls,msg:str):
            pass
        @classmethod
        def info(cls,msg:str):
            pass
        
        @classmethod
        def warning(cls,msg:str):
            pass
        @classmethod
        def error(cls,msg:str):
            pass
        @classmethod
        def critical(cls,msg:str):
            
            pass
        def setLevel(self,level:int):
            pass
        
def get_logger(
    name:str,
    if_enable_logging:bool,
    if_show_on_terminal:bool = True,
    if_save_to_disk:bool = True,
    output_root_dir:str = '.',
    terminal_leval:str = 30,
    save_leval :str = 10,
    save_folder_name:str = 'custom_log',
    fmt_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    
    fmt = logging.Formatter(fmt_string)
    
    if not if_enable_logging:
        no_logger = NO_Logging_logger()
        return no_logger
    
    
    else:
        logger = logging.getLogger(name)
        if if_show_on_terminal:
            handler_stream = logging.StreamHandler()
            handler_stream.setLevel(terminal_leval)
            handler_stream.setFormatter(fmt)
            logger.addHandler(handler_stream)
            
        if if_save_to_disk:
            
            save_folder = os.path.join(output_root_dir,save_folder_name)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_log_name = os.path.join(save_folder,f"{name}_{current_time}.log")
            
            handler_saver = logging.FileHandler(save_log_name,"w",'utf-8')
            handler_saver.setLevel(save_leval)
            handler_saver.setFormatter(fmt)
            logger.addHandler(handler_saver)
        logger.setLevel(logging.DEBUG)
        
        return logger



