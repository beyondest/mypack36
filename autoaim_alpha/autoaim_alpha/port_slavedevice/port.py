from .com_tools import *
from ..os_op.basic import *
from ..os_op.global_logger import *







class Port_Params(Params):
    
    def __init__(self) -> None:
        super().__init__()
        self.port_abs_path = '/dev/ttyUSB0'
        self.bitesize =8
        self.baudrate = 115200
        self.action_data = action_data()
        self.syn_data = syn_data()
        self.pos_data = pos_data()
        
        


class Port:
    def __init__(self,
                 mode:str = 'Dbg',
                 port_config_yaml_path:Union[str,None]=None) -> None:
        CHECK_INPUT_VALID(mode,['Dbg','Rel'])
        self.params = Port_Params()
        if port_config_yaml_path is not None:
            self.params.load_params_from_yaml(port_config_yaml_path)

        self.ser = serial.Serial(
                                port=self.params.port_abs_path,
                                baudrate=self.params.baudrate,
                                bytesize=self.params.bitesize,
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
        
        
    def send_decision(self,decision:dict):
        
        if decision['type'] == 'action':
            self.params.action_data.target_pitch_10000 = int(decision['required_pitch'] * 10000)
            self.params.action_data.target_yaw_10000 = int((decision['required_yaw'] - np.pi) * 10000)
            self.params.action_data.target_minute = int(decision['required_time_minute'])
            self.params.action_data.target_second = int(decision['required_time_second'])
            self.params.action_data.target_second_frac_10000 = int(decision['required_time_second_frac'] * 10000)
            self.params.action_data.fire_times = int(decision['required_fire_times'])
        
            msg = self.params.action_data.convert_action_data_to_bytes(if_part_crc=False)
            send_data(self.ser,msg)
            
        elif decision['type'] =='syn':
            self.params.syn_data.present_minute = int(decision['required_time_minute'])
            self.params.syn_data.present_second = int(decision['required_time_second'])
            self.params.syn_data.present_second_frac_10000 = int(decision['required_time_second_frac'] * 10000)
            
            msg = self.params.syn_data.convert_syn_data_to_bytes(if_part_crc=False)
            send_data(self.ser,msg)
        
        
    def recv_feedback(self)->tuple:
        """_summary_

        Returns:
            if_error:bool
            cur_yaw:float
            cur_pitch:float
            cur_time_minute:int
            cur_time_second:int
            cur_time_second_frac:float
        """
        
        
        msg = read_data(self.ser,
                        16)
        if_error = self.params.pos_data.convert_pos_bytes_to_data(msg,if_part_crc=False)
        
        cur_yaw = self.params.pos_data.present_yaw + np.pi
        cur_pitch = self.params.pos_data.present_pitch
        cur_time_minute = self.params.pos_data.stm_minute
        cur_time_second = self.params.pos_data.stm_second
        cur_time_second_frac = self.params.pos_data.stm_second_frac 
        
        return if_error,cur_yaw,cur_pitch,cur_time_minute,cur_time_second,cur_time_second_frac
    
    
    def port_open(self):
        self.ser.open()
    
    def port_close(self):
        self.ser.close()
        
    