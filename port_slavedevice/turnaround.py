from gimbaltest_pack import *
from serial import Serial
serialPort = '/dev/ttyTHS0'  
baudRate = 115200  

if __name__=='__main__':
    ser=Serial(serialPort,baudRate,timeout=0.5)
    turn_around()
    

