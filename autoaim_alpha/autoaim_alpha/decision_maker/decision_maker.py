from .ballistic_predictor import *
from .observer import *
from ..os_op.basic import *
from ..os_op.global_logger import *


class Decision_Maker_Params(Params):
    def __init__(self) -> None:
        super().__init__()
        


        self.cur_yaw = 0.0
        self.cur_pitch = 0.0
        self.remaining_health = 0.0
        self.remaining_ammo = 0
        self.electric_system_zero_unix_time = 0
        self.electric_system_unix_time = 0  
        
        
        self.fire_mode = 1
        """fire mode:
            0: firepower priority
            1: accuracy priority
            2: fixed angle interval shooting
        """
        

class Decision_Maker:
    def __init__(self,
                 mode:str,
                 decision_maker_params_yaml_path:Union[str,None] = None,
                 enemy_car_list:list = None
                 ) -> None:
        CHECK_INPUT_VALID(mode,"Dbg",'Rel')
        if enemy_car_list is None:
            lr1.critical("enemy_car_list is None")
            
        self.mode = mode
        self.params = Decision_Maker_Params()
        self.enemy_car_list = enemy_car_list
        if decision_maker_params_yaml_path is not None:
            self.params.load_params_from_yaml(decision_maker_params_yaml_path)
            
        self.armor_state_predict_list = [Armor_Params(enemy_car['armor_name'],armor_id) \
                                                        for enemy_car in self.enemy_car_list \
                                                            for armor_id in range(enemy_car['armor_nums'])]
        
        

        
    def update_enemy_side_info(self,
                      armor_name:str,
                      armor_id:int,
                      armor_tvec:np.ndarray,
                      armor_rvec:np.ndarray,
                      armor_confidence:float = 0.5,
                      armor_time:float = 0.0)->None:
        
        for armor_params in self.armor_state_predict_list:
            
            if armor_params.name == armor_name and armor_params.id == armor_id:
                armor_params.tvec = armor_tvec
                armor_params.rvec = armor_rvec
                armor_params.confidence = armor_confidence
                armor_params.time = armor_time
                
                
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

    def choose_target(self)->Armor_Params:
        
        nearest_armor_params = min(self.armor_state_predict_list, key=lambda x: x.tvec[2])
        if self.mode == 'Dbg':
            pass
            #for armor_params in self.armor_state_predict_list:
                #lr1.debug(f"Decision_Maker : For chosen : armor {armor_params.name} id {armor_params.id} : tvec {armor_params.tvec}, t : {armor_params.time}")
                
        lr1.debug(f"Decision_Maker : Choosed nearest_armor_state: {nearest_armor_params.name} id {nearest_armor_params.id} : tvec {nearest_armor_params.tvec}, t : {nearest_armor_params.time}")
        
        return nearest_armor_params
    
    def save_params_to_yaml(self,yaml_path:str)->None:
        self.params.save_params_to_yaml(yaml_path)
        