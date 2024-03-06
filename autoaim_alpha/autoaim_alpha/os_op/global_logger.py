import sys
sys.path.append('..')
from .mylog import *


DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50
NO_LOG = 60

lr1 = get_logger('logger1',
                 if_enable_logging=True,
                 if_show_on_terminal=True,
                 if_save_to_disk=True,
                 terminal_leval=WARNING,
                 save_leval=DEBUG,
                 save_folder_name='/home/liyuxuan/.ros/log/custom_log'
                 )


