from ser import Serial_communication
import time
import math
from serial import Serial
serialPort = '/dev/ttyTHS0'  
baudRate = 115200  

pi=math.pi

def turn_around(one_cycle_time_s:int=10,
                fps:int=10,
                yaw_start=0,
                yaw_end=0,
                pitch_start=0,
                pitch_end=0,
                begin_sleep_time_s=2,
                mode:str='yaw',
                ser:Serial=1
                ):
    times=one_cycle_time_s*fps
    each=2*pi/times
    sleep_time=1/fps
    
    #init
    Serial_communication(yaw_start,pitch_start,fps,ser=ser)
    time.sleep(begin_sleep_time_s)
    #turn
    for i in range(times):
        if mode=='yaw':
            yaw_start+=each
        elif mode=='pitch':
            pitch_start+=each
        Serial_communication(yaw_start,pitch_start,fps,ser=ser)
        time.sleep(sleep_time)
    #uninit
    Serial_communication(yaw_end,pitch_end,fps,ser=ser)
    
    


    
    #init
    Serial_communication(yaw_start,pitch_start,fps,ser=ser)
    time.sleep(begin_sleep_time_s)
    #turn
    for i in range(times):
        if mode=='yaw':
            yaw_start+=each
        elif mode=='pitch':
            pitch_start+=each
        Serial_communication(yaw_start,pitch_start,fps,ser=ser)
        time.sleep(sleep_time)
    #uninit
    Serial_communication(yaw_end,pitch_end,fps)
    
    
    
def loc_test(yaw=0.79,pitch=0.79,fps:int=20,ser=-1):
    """
    0.79->pi/4\n
    0.39->pi/8\n
    will send same message 5 times in 2.5 seconds
    
    
    """
    for i in range(5):
        Serial_communication(yaw,pitch,fps,ser=ser)
        time.sleep(0.5)
       
       
if __name__=='__main__':
    try:
        ser=Serial(serialPort,baudRate,timeout=0.5)
        print(f'serial init {serialPort} {baudRate} successfully')
    except:
        raise TypeError(f"serial init failed {serialPort} {baudRate}")
    loc_test(pitch=0,yaw=0.395,ser=ser)
    loc_test(yaw=-0.395,pitch=0.2,ser=ser) 


        
        