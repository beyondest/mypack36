from mylog import *
import sys
sys.path.append('..')
from os_op.decorator import *


a = get_logger('fucka',False,terminal_leval=30,save_leval=10)

@timing(10000,if_show_total=True)
def run():
    global a


    a.info('hello')
    a.debug('fuck')
    a.error('oh')
    a.warning('sh')
    a.critical('shit')

    return None

_,t = run()
print(f"{t:.6f}")
