import struct
import numpy as np
from crcmod.predefined import mkPredefinedCrcFun
import serial

# Integer types:
# 'b': signed byte (1 byte)                 -128-127
# 'B': unsigned byte (1 byte)               0-255
# 'h': short integer (2 bytes)              -32768-32767
# 'H': unsigned short integer (2 bytes)     0-65535
# 'i': integer (4 bytes)                    
# 'I': unsigned integer (4 bytes)
# 'l': long integer (4 bytes)
# 'L': unsigned long integer (4 bytes)
# 'q': long long integer (8 bytes)
# 'Q': unsigned long long integer (8 bytes)

# Floating-point types:
# 'f': single-precision float (4 bytes)
# 'd': double-precision float (8 bytes)

# Character type:
# 'c': character (1 byte)   (notice that struct.pack will raise error when use ('c','A'),you should use ('B',ord('A') instead))

# String type:
# 's': string (must be followed by a number, e.g., '4s' means a string of length 4)

# Special type:
# '?': boolean (1 byte)

# Padding:
# 'x': padding byte

# Byte order:
# '<': little-endian
# '>': big-endian
# '!': network byte order (big-endian)


def send_data(ser:serial.Serial,msg:bytes)->None:
    if ser.is_open:
        if not isinstance(msg,bytes):

            raise TypeError('send_data msg wrong format')
        
        ser.write(msg)
    else:
        raise ValueError('serial port not open')
        

def read_data(ser:serial.Serial,byte_len:int = 16)->bytes:
    """read data from port:ser

    Args:
        ser (serial.Serial): _description_

    Returns:
        bytes: ori_byte
    """
    if ser.is_open:
        #10 byte works well
        com_input = ser.read(byte_len)
        return com_input
    else:
        raise ValueError('serial port not open')
           
def port_close(ser:serial.Serial):
    ser.close()          
   
def port_open(port_abs_path:str = '/dev/ttyUSB0',
              bytesize:int = 8,
              baudrate = 9600)->serial.Serial:
    """ Change port init settings here\n
        Init settings:\n
        port=port_abs_path,\n
        baudrate=baudrate,\n
        bytesize=bytesize,\n
        parity=serial.PARITY_NONE,\n
        stopbits=serial.STOPBITS_ONE,\n
        xonxoff=False,\n
        rtscts=False,\n
        write_timeout=1,\n
        dsrdtr=None,\n
        inter_byte_timeout=0.1,\n
        exclusive=None # 1 is ok for one time long communication\n
    """
    
    ser = serial.Serial(
        port=port_abs_path,
        baudrate=baudrate,
        bytesize=bytesize,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        xonxoff=False,
        rtscts=False,
        write_timeout=1,
        dsrdtr=None,
        inter_byte_timeout=0.1,
        exclusive=None, # 1 is ok for one time long communication, but None is not, I dont know why!!!!
        timeout=1
    ) 
    
    return ser
    

def convert_to_bytes(value_tuple:tuple)->bytes:
    """(value,conver_type_str)
    e.g. (3.14,'<f')

    Args:
        value_tuple (tuple): _description_
    """
    if value_tuple[1][-1] == 'c':
        out = struct.pack(value_tuple[1][0]+'B',ord(value_tuple[0]))
    else:
        out = struct.pack(value_tuple[1],value_tuple[0])
    
    return out

def convert_to_data(byte_tuple:tuple):
    
    return struct.unpack(byte_tuple[1],byte_tuple[0])



def cal_crc(data:bytes)->int:
    """HOW TO GET RIGHT CRC:\n
       Input byte order same as memory in stm32\n, e.g.:0x44434241---b'ABCD'
       Make inverse of each 4 bytes of all_bytes, e.g.:b'abcd1234'----b'dcba4321'
       explain stm32 crc byte in little endian mode\n
       -> right result

    Args:
        data (bytes): _description_

    Raises:
        TypeError: _description_

    Returns:
        int: _description_
    """
    if not isinstance(data,bytes):
        raise TypeError('cal_crc must input bytes type data')
    crc_value = mkPredefinedCrcFun('crc-32-mpeg')(data)
    
    return crc_value



#upper part:   
'''
Action data: send | SOF: 'A'
By present_pitch and present_yaw send by stm32, calculte RELATIVE pitch and yaw angle and best reach time
By bollets left and some other info , decide the fire mode and fire times(default: fire x times from stm32 get msg to reach target pos, right closed)

Syn data: send | SOF: 'S'
Get present time minute and seconds and second.frac4*10000, send to stm32

Pos data: receive |SOF: 'P'
Receive present_yaw and present_pitch and present time from stm32 
'''
#stm32 part:
'''

Action data: receive |SOF : 'A'
By RELATIVE pitch and yaw sent by upper and reach target time, control 6020 to target pos on time
By fire times and reach target time, calculate uinform distributed firing time points within this time interval, and fire at each firing time point

Syn data: receive | SOF : 'S'
Once get present time from upper, stm32 will set RTC module to SYNCHRONIZE with upper pc
Once time gap between stm32 and upper is bigger than threshold , stm32 correct RTC or set bias by software code

Pos data: send | SOF : 'P'
By info from 6020 , send present yaw and pitch to upper
By info from RTC , send present time minute and second and second_frac4*10000 to upper
'''
'''
stm32 send format:


'''

class data_list:
    def __init__(self) -> None:
        """If use this as parent class, must cover len and list and labellist

        Args:
            length (_type_): _description_
        """
        self.count = 0
        self.len = 0
        self.list =[]
        self.label_list =[]
        self.crc_v = 0
        self.error = False
    def __iter__(self):
        return self
    def __len__(self):
        return self.len
    
    def __next__(self):
        if self.count < self.len:
            result = self.list[self.count]
            self.count+=1
            return result
        else:
            self.count = 0
            raise StopIteration
        
    def __getitem__(self,key):
        return self.list[key]
    
    def show(self):
        for i in range(self.len):
            print(f'{self.label_list[i]} = {self.list[i]}')
        
    def flip_4bytes_atime(self,bytes_12:bytes)->bytes:
        """return for_crc_cal, bytes_12

        Args:
            bytes_12 (bytes): _description_

        Raises:
            TypeError: _description_
            TypeError: _description_

        Returns:
            bytes: _description_
        """
        if not isinstance(bytes_12,bytes):
            raise TypeError('wrong input type, must be bytes')
        if len(bytes_12) != 12:
            raise TypeError('wrong bytes length, must be 16')
        out = list(bytes_12)
        out[0:4] = out[0:4][::-1]
        out[4:8] = out[4:8][::-1]
        out[8:12] = out[8:12][::-1]
        return bytes(out)

class syn_data(data_list):
    def __init__(self,
                 SOF:str = 'S',
                 present_minute:int=20,
                 present_second:int=30,
                 present_second_frac_10000:int=1234) -> None:
        """Null bytes will generate by convert_to_byte func
           

        Args:
            SOF (str, optional): _description_. Defaults to 'S'.
            present_minute (int, optional): _description_. Defaults to 0.
            present_second (int, optional): _description_. Defaults to 0.
            present_second_frac_10000 (int, optional): _description_. Defaults to 0.
        """
        super().__init__()
        
        self.SOF = SOF
        self.present_minute = present_minute
        self.present_second = present_second
        self.present_second_frac_10000 = present_second_frac_10000
        self.list = [self.SOF,self.present_minute,self.present_second,self.present_second_frac_10000]
        
        self.len = len(self.list)
        
        self.label_list = ['SOF','pre_min','pre_sec','pre_sec_frac']

    def convert_syn_data_to_bytes(self,if_crc:bool = True,if_revin_crc:bool = True, if_part_crc :bool =True)->bytes:
        """Calculate crc here if needed
        NO.0 (SOF:char , '<c')                             |     ('S')                      |byte0      bytes 1     total 1
        NO.1 (present_time_minute:int , '<B')              |     (0<=x<60)                  |byte1      bytes 1     total 2   
        NO.2 (present_time_second:int , '<B')              |     (0<=x<60)                  |byte2      bytes 1     total 3
        NO.3 (present_time_second_frac.4*10000:int, '<H')  |     (0<=x<=10000)              |byte3-4    bytes 2     total 5
        NO.4 (null_byte:b'1234567')   (auto add)(must add) |                                |byte5-11   bytes 7     total 12
        NO.5 (crc_value:int , '<I')  (auto add to end)     |     (return of cal_crc func)   |byte12-15  bytes 4     total 16
        PART_CRC:1-4 byte
        ALL: 6 elements,list has 4 elements
        """
        self.list = [self.SOF,self.present_minute,self.present_second,self.present_second_frac_10000]
        fmt_list = ['<c','<B','<B','<H']
        out = b''
        crc_v =b''
        null_bytes = b'1234567'
        
        for index,each in enumerate(self.list):
            out += convert_to_bytes((each,fmt_list[index]))
            
            
        out+=null_bytes
        if if_crc:
            if if_revin_crc:
                
                if if_part_crc:
                    for_crc_cal = out[1:5]
                    for_crc_cal = for_crc_cal[::-1]
                    crc_v = cal_crc(for_crc_cal)
                else:
                    for_crc_cal=self.flip_4bytes_atime(out)
                    crc_v = cal_crc(for_crc_cal)
            else:
                raise TypeError("This function isnot support yet")     
            self.crc_v = crc_v
            crc_v = convert_to_bytes((crc_v,'<I'))
        else:
            crc_v =b''
        out+= crc_v
        return out   
        
            
class action_data(data_list):
    
    def __init__(self,
                 SOF:str = 'A',
                 fire_times:int=0,
                 abs_pitch_10000:int=+1745, # -1745 = -10 degree
                 abs_yaw_10000:int=-15708,  # 15708 = 90 degree
                 target_minute:int=0,
                 target_second:int=0,
                 target_second_frac_10000:int=0,
                 reserved_slot:int=0) -> None:
        
        super().__init__()
        
        self.label_list = ['SOF','ftimes','tarpitch','taryaw','tarmin','tarsec','tarsecfrac','svolrpm']
        self.SOF = SOF
        self.fire_times = fire_times
        self.abs_pitch_10000 = abs_pitch_10000
        self.abs_yaw_10000 = abs_yaw_10000
        self.target_minute = target_minute
        self.target_second = target_second
        self.target_second_frac_10000 = target_second_frac_10000
        self.reserved_slot = reserved_slot
        self.list = [self.SOF,
                     self.fire_times,
                     self.abs_pitch_10000,
                     self.abs_yaw_10000,
                     self.target_minute,
                     self.target_second,
                     self.target_second_frac_10000,
                     self.reserved_slot]
        self.len = len(self.list)
        
    def convert_action_data_to_bytes(self,if_crc:bool = True, if_revin_crc:bool = True , if_part_crc:bool = True)->bytes:
        """Calculate crc here if needed
        NO.0 (SOF:char , '<c')                             |     ('A')                      |byte0      bytes 1     total 1
        NO.1 (fire_times:int , '<b')                       |     (-1<=x<=100)               |byte1      bytes 1     total 2 (-1:not control;0:control not fire) 
        NO.2 (target_pitch.4*10000:int , '<h')             |     (abs(x)<=15708)            |byte2-3    bytes 2     total 4
        NO.3 (target_yaw.4*10000:int , '<h')               |     (abs(x)<=31416)            |byte4-5    bytes 2     total 6
        NO.4 (reach_target_time_minute:int , '<B')         |     (0<=x<60)                  |byte6      bytes 1     total 7
        NO.5 (reach_target_time_second:int , '<B')         |     (0<=x<=60)                 |byte7      bytes 1     total 8
        NO.6 (reach_target_time_second_frac.4*10000 , '<H')|     (0<=x<=10000)              |byte8-9    bytes 2     total 10 
        NO.78(reserved_slot:int, '<h')(only for debug)| (-30000<=x<=30000 if vol)  |byte10-11  bytes 2     total 12
        NO.9(crc_value:int , '<I')   (auto add to end)     |     (return of cal_crc func)   |byte12-15  bytes 4     total 16
        
        PART_CRC: byte2-5
        ALL: 10 elements ,list has 7 elements
        """
        self.list = [self.SOF,
                     self.fire_times,
                     self.abs_pitch_10000,
                     self.abs_yaw_10000,
                     self.target_minute,
                     self.target_second,
                     self.target_second_frac_10000,
                     self.reserved_slot]
        
        fmt_list = ['<c','<b','<h','<h','<B','<B','<H','<h']
        out = b''
        crc_v =b''
        
        for index,each in enumerate(self.list):
            out += convert_to_bytes((each,fmt_list[index]))
        
        if if_crc:
            if if_revin_crc:
                if if_part_crc:
                    for_crc_cal = out[2:6]
                    for_crc_cal = for_crc_cal[::-1]                    
                    crc_v = cal_crc(for_crc_cal)
                else:
                    for_crc_cal = self.flip_4bytes_atime(out)
                    crc_v = cal_crc(for_crc_cal)
            else:
                raise TypeError('this function not support yet')
            self.crc_v = crc_v
            
            crc_v = convert_to_bytes((crc_v,'<I'))
            
        else:
            crc_v = b''
              
        out+= crc_v
        return out  


class pos_data(data_list):
    
    def __init__(self,
                 SOF:str = 'P',
                 stm_minute:int = 20,
                 stm_second:int = 30,
                 stm_second_frac:float = 0.1234,
                 present_pitch:float = -0.1745,
                 present_yaw:float = -0.1745,
                 present_debug_value:int = -1
                 ) -> None:
        super().__init__()
        
        self.label_list = ['SOF','stmin','stsec','stsecfrac','prepit','preyaw','predbgval']
        self.SOF =SOF
        self.stm_minute = stm_minute
        self.stm_second = stm_second
        self.stm_second_frac = stm_second_frac
        self.present_pitch = present_pitch
        self.present_yaw = present_yaw
        self.present_debug_value = present_debug_value
        self.error =False
        self.crc_get =0
        self.list = [self.SOF,
                     self.stm_minute,
                     self.stm_second,
                     self.stm_second_frac,
                     self.present_pitch,
                     self.present_yaw,
                     self.present_debug_value]
        self.len = len(self.list)
        
        

    def convert_pos_bytes_to_data(self,ser_read:bytes,if_crc:bool = True,if_crc_rev:bool = True,if_part_crc:bool = True)->bool:
        """Convert what STM32 send, Save data to self.list
        NO.0 (SOF:char , '<c')                             |     ('P')                      |byte0      bytes 1     total 1
        NO.1 (present_time_minute:int , '<B')              |     (0<=x<60)                  |byte1      bytes 1     total 2   
        NO.2 (present_time_second:int , '<B')              |     (0<=x<60)                  |byte2      bytes 1     total 3
        NO.3 (present_time_second_frac.4*10000:int, '<H')  |     (0<=x<=10000)              |byte3-4    bytes 2     total 5
        NO.4 (present_pitch.4*10000:int , '<h')            |     (abs(x)<=15708)            |byte5-6    bytes 2     total 7
        NO.5 (present_yaw.4*10000:int , '<h')              |     (abs(x)<=31416)            |byte7-8    bytes 2     total 9
        NO.6 (present_debug_value:rpm or torque I,'<h')    |                                |byte9-10   bytes 2     total 11
        NO.7 (nullbyte: char='1','<c')                     |                                |byte11     bytes 1     total 12
        NO.8 (crc_value:int , '<I')                        |     (return of cal_crc func)   |byte12-15  bytes 4     total 16
        PART_CRC: byte 5-8
        ALL: 9 elements , list has 7 elements
        Return:
            if_error
        """
        self.fmt_list = ['<c','<B','<B','<H','<h','<h','<h','<c','<I']
        self.range_list = [(0,1),(1,2),(2,3),(3,5),(5,7),(7,9),(9,11),(11,12),(12,16)]
        self.frame_type_nums = 9
        out = []
        error = False
        for i in range(self.frame_type_nums):
            out.append(struct.unpack(self.fmt_list[i],
                                    ser_read[self.range_list[i][0]:self.range_list[i][1]]
                                    )[0]
                    )
        
        out[0] = out[0].decode('utf-8')
        out[7]= out[7].decode('utf-8')
        
            
        if out[0] == self.SOF:
            if if_crc:
                if if_crc_rev:
                    if if_part_crc:
                        for_crc_cal = ser_read[5:9]
                        for_crc_cal = for_crc_cal[::-1]
                        crc_v = cal_crc(for_crc_cal)
                        error = not (crc_v == out[8])
                    else:
                        for_crc_cal = ser_read[0:12]
                        for_crc_cal = self.flip_4bytes_atime(for_crc_cal)
                        crc_v = cal_crc(for_crc_cal)
                        error = not (crc_v == out[8])
                else:
                    raise TypeError("This function is not support yet")
            else:
                crc_v = 0  
                error = False 
        else:
            error = True
        
        

        self.error =error
        self.crc_v = crc_v
        self.crc_get = out[8]
          
        self.SOF = out[0]
        self.stm_minute = out[1]
        self.stm_second = out[2]
        self.stm_second_frac = round(out[3]/10000,4)
        self.present_pitch = round(out[4]/10000,4)
        self.present_yaw = round(out[5]/10000,4)
        self.present_debug_value = out[6]
        
        self.list = [self.SOF,
                     self.stm_minute,
                     self.stm_second,
                     self.stm_second_frac,
                     self.present_pitch,
                     self.present_yaw,
                     self.present_debug_value]
        
        
        
        return error

    

#first get reversed of data, then use crc-32-mpeg will get right crc same as stm32 
import numpy as np
if __name__ =="__main__":
    
    
    a = action_data()
    if 1:
        for yaw in np.arange(-31416,31416,2000):
            
            a.abs_yaw_10000 = int(yaw)
            
            r = a.convert_action_data_to_bytes(if_part_crc=False)

            s = ''
            for i in r:
                
                add_ = hex(i)[2:]
                if len(add_) == 1:
                    add_ = '0'+add_
                s+= add_ + ' '
                
            print(s)
        
    if 0:
        for pitch in np.arange(-3491,8727,1000):
            
            a.abs_pitch_10000 = int(pitch)
            
            r = a.convert_action_data_to_bytes(if_part_crc=False)

            s = ''
            for i in r:
                
                add_ = hex(i)[2:]
                if len(add_) == 1:
                    add_ = '0'+add_
                s+= add_ + ' '
                
            print(s)
            
    
        
        
    
    
    
    
    
    
    