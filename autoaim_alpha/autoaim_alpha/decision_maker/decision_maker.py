from .ballistic_predictor import *
from .observer import *
from ..os_op.basic import *
from ..os_op.global_logger import *


class Decision_Maker_Params(Params):
    def __init__(self) -> None:
        super().__init__()
        self.car_latest_state_list = []
        self.armor_latest_state_list = []
        self.cur_yaw = 0.0
        self.cur_pitch = 0.0
        self.remaining_health = 0.0
        self.remaining_ammo = 0
        self.electric_system_zero_unix_time = 0
        self.electric_system_unix_time = 0  
        
        """fire mode:
            0: firepower priority
            1: accuracy priority
            2: fixed angle interval shooting
        """
        self.fire_mode = 1
        
        

class Decision_Maker:
    def __init__(self,
                 mode:str,
                 decision_maker_params_yaml_path:Union[str,None]
                 ) -> None:
        CHECK_INPUT_VALID(mode,"Dbg",'Rel')
        self.mode = mode
        self.params = Decision_Maker_Params()
        if decision_maker_params_yaml_path is not None:
            self.params.load_params_from_yaml(decision_maker_params_yaml_path)
        
        
    def update_enemy_side_info(self,
                      car_state_list:list,
                      armor_state_list:list
                      ):
        self.params.car_latest_state_list = car_state_list
        self.params.armor_latest_state_list = armor_state_list
        
    
    def update_our_side_info(self,
                             cur_yaw:float,
                             cur_pitch:float,
                             electric_system_minute:int,
                             electric_system_second:int,
                             electric_system_second_frac:float,
                             remaining_health:Union[float,None] = None,
                             remaining_ammo:Union[float,None] = None,
                             fire_mode:Union[int,None] = None)->None:
        
        self.params.cur_yaw = cur_yaw
        self.params.cur_pitch = cur_pitch
        self.params.electric_system_unix_time = trans_t_to_unix_time(electric_system_minute,
                                                                     electric_system_second,
                                                                     electric_system_second_frac,
                                                                     self.params.electric_system_zero_unix_time)      
        
        
        if remaining_health is not None:
            self.params.remaining_health = remaining_health
        if remaining_ammo is not None:
            self.params.remaining_ammo = remaining_ammo
        if fire_mode is not None:
            self.params.fire_mode = fire_mode    

    def make_decision(self)->dict:
        """

        Returns:
            dict: 
                type: 'action' or'syn'
                required_yaw: float
                required_pitch: float
                required_fire_times: int
                syn_time_minute: int
                syn_time_second: int
                syn_time_second_frac: float
        """
        
        decision = {}
        required_yaw = ...
        required_pitch = ...
        required_fire_times = ...
        required_unix_time = ...
        
        
        required_time_minute, required_time_second, required_time_second_frac = trans_unix_time_to_t(self.params.electric_system_zero_unix_time,
                                                                                                     required_unix_time)
        
        decision['type'] = 'action' # action or syn
        decision['required_yaw'] = required_yaw
        decision['required_pitch'] = required_pitch
        decision['required_fire_times'] = required_fire_times
        decision['required_time_minute'] = required_time_minute
        decision['required_time_second'] = required_time_second
        decision['required_time_second_frac'] = required_time_second_frac
        
        
        return decision
        