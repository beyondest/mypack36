from ..os_op.basic import *
from ..os_op.global_logger import *
import numpy as np
import time
from .tools import *


class Score_Obj_for_automatic_matching(Score_Obj):
    def __init__(self,
                 name:str,
                 attributes_list:list,
                 score_accumulation_method_list:list,
                 weights_list:list,
                 reference_list:list):
        
        """attribute_id (int):
                0: armor_name:str
                1: tvec_at_specific_time:np.ndarray, (3,)
                2: rvec_at_specific_time:np.ndarray, (3,)
                3: armor_confidence:float
            

        Args:
            name (str): _description_
            attributes_list (list): _description_
            score_accumulation_method_list (list): _description_
            reference_list (list): _description_
        """
        super().__init__(name,
                         attributes_list,
                         score_accumulation_method_list,
                         weights_list,
                         reference_list)
        
        
    
    def turn_attribute_to_score(self, attribute_id, attribute_value, reference_value: float | None = None) -> np.float:
        
        """
        Returns:
            float: _description_
        """
        
        if attribute_id == 0:
            return attribute_value == reference_value 
        
        if attribute_id == 1 :
            dis_criterion = 1
            min_dis = 0.01
            dis = np.linalg.norm(attribute_value - reference_value)
            if dis < min_dis:
                dis = min_dis
            return dis_criterion/dis
        
        if attribute_id == 2 :
            dis_criterion = 1
            min_dis = 0.01
            dis = np.linalg.norm(attribute_value - reference_value)
            if dis < min_dis:
                dis = min_dis
            return dis_criterion/dis
        
        if attribute_id == 3 :
            return attribute_value
       

class Armor_Params(Params):
    
    def __init__(self,
                 name:str,
                 id :int
                 ):
        super().__init__()
        
        self.tvec = np.zeros(3)         # in camera frame
        self.rvec = np.zeros(3)         # in camera frame
        self.tvec_in_car_frame = np.zeros(3)   # in car frame
        self.rvec_in_car_frame = np.zeros(3)   # in car frame
        
        self.time = 0.0     
        self.confidence = 0.0  
        self.id = id 
        self.name = name 
        


class Car_Params(Params):
    def __init__(self,
                 armor_distance:list,
                 armor_distance_variation:list,
                 armor_name:str,
                 Q:np.ndarray,
                 R:np.ndarray,
                 H:np.ndarray,
                 history_depth:int=10,
                 armor_nums:int=4,
                 ):
        """Armor id:(counterclockwise)
            0: front
            1: right
            2: back
            3: left
            
          or 
            0: front
            1: back

        Args:
            armor_distance_in_x_axis (float): _description_
            armor_name (str, optional): _description_. Defaults to '3x'.
            history_depth (int, optional): _description_. Defaults to 10.
            armor_nums (int, optional): _description_. Defaults to 4.
        """
        
        super().__init__()
        
        self.armor_distance = armor_distance
        self.armor_distance_variation = armor_distance_variation
        
        self.armor_name = armor_name
        self.history_depth = history_depth
        self.armor_nums = armor_nums
        
        self.car_tvec_history_list = [np.zeros(3) for i in range(self.history_depth)]   # history of car center position in camera frame
        self.car_tv_vec_history_list = [np.zeros(3) for i in range(self.history_depth)]   # history of car center velocity in camera frame
        self.car_ta_vec_history_list = [np.zeros(3) for i in range(self.history_depth)]   # history of car center acceleration in camera frame
        self.car_rvec_history_list = [np.zeros(3) for i in range(self.history_depth)]   # history of armor_id 0 rotation position in camera frame
        self.car_rv_vec_history_list = [np.zeros(3) for i in range(self.history_depth)]   # history of armor_id 0 rotation velocity in camera frame
        self.car_ra_vec_history_list = [np.zeros(3) for i in range(self.history_depth)]   # history of armor_id 0 rotation acceleration in camera frame
        self.car_rotation_speed_history_list = [0.0 for i in range(self.history_depth)]   # history of car rotation speed in car frame
        
        self.car_confidence_history_list = [0.0 for i in range(self.history_depth)]     # history of all armors confidence
        self.car_time_history_list = [0.0 for i in range(self.history_depth)]     # history of car time      

        self.armor_idx_to_predict_history = {}
        self.armor_idx_to_detect_history = {}
        self.armor_idx_to_tvec_kf = {}
        self.armor_idx_to_rvec_kf = {}
        
        for i in range(self.armor_nums):
            
            self.armor_idx_to_predict_history.update({i :  [Armor_Params(armor_name,i) for j in range(self.history_depth)]})
            self.armor_idx_to_detect_history.update({i :  [Armor_Params(armor_name,i) for j in range(self.history_depth)]})
            self.armor_idx_to_tvec_kf.update({i :  Kalman_Filter(Q,R,H)})
            self.armor_idx_to_rvec_kf.update({i :  Kalman_Filter(Q,R,H)})
            
            
class Observer_Params(Params):
    
    def __init__(self):
        
        super().__init__()
        
        # armor_distance : [x_distance, y_distance], x distance means maximum efficency direction of robot chassis
        self.enemy_car_list = [{'armor_name':'3x','armor_distance':[0.4,0.5],'armor_nums': 4},
                               {'armor_name':'4x','armor_distance': [0.5,0.5],'armor_nums': 2}]
        
        self.Q_scale = 0.1
        self.R_scale_dis_diff_denominator = 0.1 # unit: meter; R_scale = dis_diff / R_scale_dis_diff_denominator
        
        self.H = np.eye(3)
        
        # max distance variation of armor distance in x and y axis, if 0, forbid armor distance change
        self.armo_distance_variation = [0.05,0.05]
        
        self.history_depth = 10
        self.predict_frequency = 10
        self.if_force_predict_after_detect = 1
        self.min_continous_num_to_apply_predict = 3
        self.min_dis_between_continous_detection = 0.1 # unit: meter
        self.min_time_between_continous_detection = 0.1 # unit: second 
        self.min_predict_period = 0.1 # unit: second 
        self.min_continous_num_to_predict_armor = 3
        
        # [score_weight_for_armor_name, score_weight_for_tvec, score_weight_for_rvec, score_weight_for_confidence]
        self.score_weights_list_for_automatic_matching = [100,10,10,10]
        self.score_accumulation_method_list_for_automatic_matching = ['add','add','add','add']
        
        # confidence = score/score_criterion
        self.score_criterion_for_automatic_matching = 200
        
        self.armor_name_to_car_params = {}
        
        
    def init_each_car_params(self):
        
        for i in self.enemy_car_list:
            if i['armor_name'] in self.armor_name_to_car_params.keys():
                lr1.critical(f"Observer: armor_name {i['armor_name']} already exists in car_params")
                raise ValueError("armor_name already exists in car_params")
                
                
            dic = {i['armor_name']:Car_Params(i['armor_distance'],
                                              self.armo_distance_variation,
                                              i['armor_name'],
                                              self.Q_scale * np.eye(3),
                                              np.eye(3),
                                              self.H,
                                              self.history_depth,
                                              i['armor_nums'])}
            
            self.armor_name_to_car_params.update(dic)
            lr1.info(f"Observer: Init {i['armor_name']} car params")
        
        lr1.info(f"Observer: All Enemy Car Nums: {len(self.enemy_car_list)}  ")
    
    
class Observer:
    
    def __init__(self,
                 mode:str = 'Dbg',
                 observer_params_yaml_path:Union[str,None]=None
                 ):
        
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        self.mode = mode
        self.observer_params = Observer_Params()
        self.latest_focus_armor_name = None
        self.latest_focus_armor_index = None
        self.latest_focus_time = None
        
        if observer_params_yaml_path is not None:
            self.observer_params.load_params_from_yaml(observer_params_yaml_path)
            self.observer_params.init_each_car_params()
        
        else:
            lr1.warning(f"Observer: observer_params_yaml_path is None, use default params")
            self.observer_params.init_each_car_params()
        
    
    def update_by_detection_list(self,
               all_targets_list:list):
        """
        Args:
            all_targets_list (list): list of all targets, each target is a dict, including:
                'armor_name':str,
                'tvec':np.ndarray, (3,)
                'rvec':np.ndarray, (3,)
                'time':float
        """
        
        for target in all_targets_list:
            self.update_by_detection(target['armor_name'],
                                       target['tvec'],
                                       target['rvec'],
                                       target['time'])
            
    def update_by_prediction_all(self):
        """
        Args:
            all_targets_list (list): list of all targets, each target is a dict, including:
                'armor_name':str
        """
        
        for target in self.observer_params.enemy_car_list:
            armor_name = target['armor_name']
            armor_nums = target['armor_nums']
            for i in range(armor_nums):
                self.update_by_prediction(armor_name, i)
            
    
    def update_by_detection(self,
                            armor_name:str,
                            tvec:np.ndarray,
                            rvec:np.ndarray,
                            time:float):
        
        
        right_armor_name,right_armor_idx,confidence = self._automatic_matching(armor_name, tvec, rvec, time)
        
        armor_nums = self.observer_params.armor_name_to_car_params[right_armor_name].armor_nums
        tvec_list,rvec_list= get_other_face_center_pos(   tvec,
                                                rvec,
                                                self.observer_params.armor_name_to_car_params[right_armor_name].armor_distance[0],
                                                self.observer_params.armor_name_to_car_params[right_armor_name].armor_distance[1],
                                                armor_nums)
        
        for i in range(armor_nums):
            armor_index = i + right_armor_idx
            if armor_index >= armor_nums:
                armor_index -= armor_nums
            detect_history_list = self.observer_params.armor_name_to_car_params[right_armor_name].armor_idx_to_detect_history[armor_index]
            self._update_armor_history_params(  detect_history_list, 
                                        tvec_list[i], 
                                        rvec_list[i], 
                                        time, 
                                        confidence)
            
        if self.mode == 'Dbg':
            lr1.info(f"Observer: Update {right_armor_name} car {right_armor_idx} armor detect_params at time {time} with confidence {confidence}")
        
        if self.observer_params.if_force_predict_after_detect:
            for i in range(armor_nums):
                self.update_by_prediction(right_armor_name, i)
            
        self.latest_focus_armor_name = right_armor_name
        self.latest_focus_armor_index = right_armor_idx
        self.latest_focus_time = time
        
        
    def update_by_prediction(self,
                              armor_name:str,
                              armor_idx:int):
        
        self._update_car_params_and_armor_relative_params()
        car_params = self.observer_params.armor_name_to_car_params[armor_name]
        detect_history_list = car_params.armor_idx_to_detect_history[armor_idx]
        predict_history_list = car_params.armor_idx_to_predict_history[armor_idx]
        
        tvec_predict, rvec_predict, cur_time, confidence = self.__cal_predict_params(detect_history_list, predict_history_list)
        self._update_armor_history_params(predict_history_list, tvec_predict, rvec_predict, cur_time, confidence)
                
        if self.mode == 'Dbg':
            
            cur_time = time.time()
            lr1.info(f"Observer: Update {armor_name} car {armor_idx} armor predict_params at time {cur_time} with confidence {confidence}") 
    
            
    def get_car_latest_state(self)->list:
        """
        
        Returns:
            list of dict:
                armor_name:str
                car_center_tvec : (armor id 0 tvec) in camera frame 
                car_center_rvec : (armor id 0 rvec) in camera frame
                car_center_tv_vec : (armor id 0 tv_vec) in camera frame
                car_rotation_speed : in car frame
                car_time:float
        """
        car_list = []
        for armor_name in self.observer_params.armor_name_to_car_params.keys():
            car_center_tvec = self.observer_params.armor_name_to_car_params[armor_name].car_tvec_history_list[0]
            car_center_rvec = self.observer_params.armor_name_to_car_params[armor_name].car_rvec_history_list[0]
            car_center_tv_vec = self.observer_params.armor_name_to_car_params[armor_name].car_tv_vec_history_list[0]
            car_rotation_speed = self.observer_params.armor_name_to_car_params[armor_name].car_rotation_speed_history_list[0] 
            car_time = self.observer_params.armor_name_to_car_params[armor_name].car_time_history_list[0]
            
            car_list.append({'armor_name':armor_name,
                             'car_center_tvec':car_center_tvec,
                             'car_center_rvec':car_center_rvec,
                             'car_center_tv_vec':car_center_tv_vec,
                             'car_rotation_speed':car_rotation_speed,
                             'car_time': car_time })
        return car_list
    
    def get_armor_latest_state(self)->list:
        """

        Returns:
            list of dict:
                armor_name:str,
                armor_id:int,
                armor_tvec :in camera frame,
                armor_rvec :in camera frame,
                armor_confidence:float = 0, 0.25, 0.5, 0.75
                armor_time:float
        """
        armor_list = []
        for armor_name in self.observer_params.armor_name_to_car_params.keys():
            for armor_id,armor_predict_history_list in self.observer_params.armor_name_to_car_params[armor_name].armor_idx_to_predict_history.items():
                armor_predict_latest_params = armor_predict_history_list[0]
                
                armor_list.append({'armor_name':armor_name,
                                   'armor_id':armor_id,
                                   'armor_tvec':armor_predict_latest_params.tvec,
                                   'armor_rvec':armor_predict_latest_params.rvec,
                                   'armor_confidence':armor_predict_latest_params.confidence,
                                   'armor_time':armor_predict_latest_params.time})

        return armor_list
    
    def predict_armor_state_by_car(self,
                                    armor_name:str,
                                    armor_index:int,
                                    specific_time:float)->list:
        """get the specific armor(id) state at specific time

        Args:
            armor_name (str): _description_
            armor_index (int): _description_
            specific_time (float): _description_

        Returns:
            armor_predict_tvec (np.ndarray): _description_
            armor_predict_rvec (np.ndarray): _description_
        """
        
        car_params = self.observer_params.armor_name_to_car_params[armor_name]
        armor_latest_params = car_params.armor_idx_to_detect_history[armor_index][0]
        
        
        period = specific_time - armor_latest_params.time
        armor_relative_tvec = armor_latest_params.tvec - car_params.car_tvec
        armor_relative_rvec = armor_latest_params.rvec - car_params.car_rvec
        
        car_predict_tvec = car_params.car_tvec + car_params.car_tv_vec * period + 0.5 * car_params.car_ta_vec * period**2
        car_predict_rvec = car_params.car_rvec + car_params.car_rv_vec * period + 0.5 * car_params.car_ra_vec * period**2
        
        armor_predict_tvec = car_predict_tvec + armor_relative_tvec
        armor_predict_rvec = car_predict_rvec + armor_relative_rvec
        
        return armor_predict_tvec, armor_predict_rvec

    def predict_armor_state_by_itself(self,
                                      armor_name:str,
                                      armor_index:int,
                                      specific_time:float):
        """get the specific armor(id) state at specific time

        Args:
            armor_name (str): _description_
            armor_index (int): _description_
            specific_time (float): _description_

        Returns:
            armor_predict_tvec (np.ndarray): _description_
            armor_predict_rvec (np.ndarray): _description_
        """
        
        predict_history_list = self.observer_params.armor_name_to_car_params[armor_name].armor_idx_to_detect_history[armor_index]
        predict_period = specific_time - predict_history_list[0].time
        
        tv_vec = (predict_history_list[0].tvec - predict_history_list[1].tvec) / predict_period
        tv_vec_old = (predict_history_list[1].tvec - predict_history_list[2].tvec) / predict_period
        ta_vec = (tv_vec - tv_vec_old) / predict_period
        
        tvec = predict_history_list[0].tvec + tv_vec * specific_time + 0.5 * ta_vec * specific_time**2
        
        rv_vec = (predict_history_list[0].rvec - predict_history_list[1].rvec) / predict_period
        rv_vec_old = (predict_history_list[1].rvec - predict_history_list[2].rvec) / predict_period
        ra_vec = (rv_vec - rv_vec_old) / predict_period
        
        rvec = predict_history_list[0].rvec + rv_vec * specific_time + 0.5 * ra_vec * specific_time**2
        
        return tvec,rvec
    
    def _automatic_matching(self,
                           detect_armor_name:str,
                           tvec:np.ndarray,
                           rvec:np.ndarray
                           )->list:
        """The armor that is closest to the given position at the given time will be selected as the target.

        Args:
            tvec (np.ndarray): _description_
            rvec (np.ndarray): _description_
            time (float): _description_

        Returns:
            armor_name:str, armor_idx:int, confidence:float
        """
        
        score_keeper = Score_Keeper()
        
        for armor_name,car_params in self.observer_params.armor_name_to_car_params.items():
            for armor_id,armor_predict_history_list in car_params.armor_idx_to_predict_history.items():
                
                armor_predict_latest_params = armor_predict_history_list[0]
                
                score_obj_name = armor_name + '_' + str(armor_id)
                attributes_list = [ armor_name,
                                    armor_predict_latest_params.tvec,
                                    armor_predict_latest_params.rvec,
                                    armor_predict_latest_params.confidence]
                reference_list =  [ detect_armor_name,
                                    tvec,
                                    rvec,
                                    None]
                
                score_obj_for_automatic_matching = Score_Obj_for_automatic_matching(score_obj_name,
                                                                                    attributes_list,
                                                                                    self.observer_params.score_accumulation_method_list_for_automatic_matching,
                                                                                    self.observer_params.score_weights_list_for_automatic_matching,
                                                                                    reference_list
                                                                                    )
                
                score_keeper.add_score_obj(score_obj_for_automatic_matching)
                
                if self.mode == 'Dbg':
                    score_obj_for_automatic_matching.show_each_attribute_score()
                    
        score_keeper.update_rank()
        best_name ,best_score = score_keeper.get_name_and_score_of_score_by_rank(0)
        right_armor_name = best_name.split('_')[0]
        right_armor_idx = int(best_name.split('_')[1])
        confidence = best_score/self.observer_params.score_criterion_for_automatic_matching
        
        if self.mode == 'Dbg':
            
            lr1.info(f"Observer :\nDetection result: {detect_armor_name} \nAutomatic matching result: {right_armor_name} {right_armor_idx} with confidence {confidence}")
        
        return right_armor_name,right_armor_idx,confidence


    def _update_armor_history_params(self,
                            armor_list:list,
                            tvec:np.ndarray,
                            rvec:np.ndarray,
                            time:float,
                            confidence:float):
        
        new_armor_params = Armor_Params(armor_list[0].id)
        new_armor_params.tvec = tvec
        new_armor_params.rvec = rvec
        new_armor_params.time = time
        new_armor_params.confidence = confidence
        
        SHIFT_LIST_AND_ASSIG_VALUE(armor_list,new_armor_params)

    def _update_car_params_and_armor_relative_params(self):
        """Will update all car params, including armor tvec,rvec in car frame too
        """
        for armor_name,car_params in self.observer_params.armor_name_to_car_params.items():
            
            tvec_list = []
            rvec_list = []
            confidence_list = []
            
            cur_time = max([armor_params.time for armor_params in car_params.armor_idx_to_predict_history[0]])
            
            period = cur_time - car_params.car_time_history_list[0]
            for armor_idx in car_params.armor_idx_to_predict_history:
                    
                latest_predict_armor_params = car_params.armor_idx_to_predict_history[armor_idx][0] 
                tvec_list.append(latest_predict_armor_params.tvec)
                rvec_list.append(latest_predict_armor_params.rvec)
                confidence_list.append(latest_predict_armor_params.confidence)
                    
            car_tvec = np.mean(np.array(tvec_list),axis=0)
            car_tv_vec = (car_tvec - car_params.car_tvec_history_list[0]) / period
            car_ta_vec = (car_tv_vec - car_params.car_tvec_history_list[0]) / period
            
            car_rvec = rvec_list[0]
            car_rv_vec = (car_rvec - car_params.car_rvec_history_list[0])/ period
            car_ra_vec = (car_rv_vec - car_params.car_rv_vec_history_list[0]) / period
            
            for armor_idx in car_params.armor_idx_to_predict_history:
                
                latest_predict_armor_params = car_params.armor_idx_to_predict_history[armor_idx][0] 
                latest_predict_armor_params.tvec_in_car_frame = latest_predict_armor_params.tvec - car_tvec
                latest_predict_armor_params.rvec_in_car_frame = latest_predict_armor_params.rvec - car_rvec
            
            car_rotation_speed = self.__cal_car_rotation_speed(armor_name)
            car_confidence = np.mean(np.array(confidence_list))
            car_time = cur_time
            
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_tvec_history_list,car_tvec)
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_tv_vec_history_list,car_tv_vec)
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_ta_vec_history_list,car_ta_vec)
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_rvec_history_list,car_rvec)
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_rv_vec_history_list,car_rv_vec)
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_ra_vec_history_list,car_ra_vec)
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_rotation_speed_history_list,car_rotation_speed)
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_confidence_history_list,car_confidence)
            SHIFT_LIST_AND_ASSIG_VALUE(car_params.car_time_history_list,car_time)
            
            if self.mode == 'Dbg':
                
                lr1.info(f"Observer: Update {armor_name} car params at time {cur_time}")
            
        if self.mode == 'Dbg':
            
            lr1.info(f"Observer: Update {armor_name} car latent_params at time {cur_time}")
            
            
    def __cal_predict_params(self,
                    armor_name:str,
                    armor_idx:int
                    )->list:
        """Use Kalman Filter to get best predict params from detect_history_list and predict_history_list

        Args:
            detect_history_list (list): _description_
            predict_history_list (list): _description_

        Returns:
            list: _description_
                tvec_predict (np.ndarray): _description_
                rvec_predict (np.ndarray): _description_
                cur_time (float): _description_
                confidence (float): 
                    if confidence == 0, means the armor is not focused, so we only use predict_history_list to predict the armor state.
                    if confidence == 0.25, continuous_num = 2
                    if confidence == 0.5, continuous_num = 3
                    if confidence == 0.75, continuous_num >3
        """
        detect_history_list = self.observer_params.armor_name_to_car_params[armor_name].armor_idx_to_detect_history[armor_idx]
        predict_history_list = self.observer_params.armor_name_to_car_params[armor_name].armor_idx_to_predict_history[armor_idx]
        
        if_focus = self.__check_if_armor_is_focused(detect_history_list[0].name,detect_history_list[0].id)
        
        cur_time = time.time()
        # only when you use _update_by_predition without _update_by_detection, this will happen
        if not if_focus:
            tvec_predict,rvec_predict = self.predict_armor_state_by_itself(armor_name,armor_idx,cur_time)
            confidence = 0
            
        # every time you use _update_by_detection, this will happen
        else:
            
            tvec_kf = self.observer_params.armor_name_to_car_params[armor_name].armor_idx_to_tvec_kf[armor_idx]
            rvec_kf = self.observer_params.armor_name_to_car_params[armor_name].armor_idx_to_rvec_kf[armor_idx]
            continous_num = self.__find_continous_num(detect_history_list)
            id = 0
            
            if continous_num == 1:
                
                tvec_init = detect_history_list[id].tvec
                P_init = np.eye(3)
                tvec_kf.set_initial_state(tvec_init,P_init)
                
                rvec_init = detect_history_list[id].rvec
                P_init = np.eye(3)
                rvec_kf.set_initial_state(rvec_init,P_init)
                
                confidence = 0
            
            
            else:
                period_new = cur_time - predict_history_list[0].time  
                
                if continous_num == 2:
                    
                    # predict tvec
                    dis = np.linalg.norm(detect_history_list[id].tvec - detect_history_list[id + 1].tvec)
                    R_scale = dis / self.observer_params.R_scale_dis_diff_denominator
                    R = np.eye(3) * R_scale
                    tv_vec = (detect_history_list[id].tvec - detect_history_list[id + 1].tvec) / (detect_history_list[id].time - detect_history_list[id + 1].time)
                    X_bias = tv_vec * period_new
                    A = np.eye(3)
                    Z = detect_history_list[id].tvec
                    tvec_kf.predict(A,Z,X_bias,None,R)

                    # predict rvec
                    dis = np.linalg.norm(detect_history_list[id].rvec - detect_history_list[id + 1].rvec)
                    R_scale = dis / self.observer_params.R_scale_dis_diff_denominator
                    R = np.eye(3) * R_scale
                    rv_vec = (detect_history_list[id].rvec - detect_history_list[id + 1].rvec) / (detect_history_list[id].time - detect_history_list[id + 1].time)
                    X_bias = rv_vec * period_new
                    A = np.eye(3)
                    Z = detect_history_list[id].rvec
                    
                    rvec_kf.predict(A,Z,X_bias,None,R)
                    
                    confidence = 0.25
                    
                elif continous_num == 3:
                    
                    # predict tvec
                    dis = np.linalg.norm(detect_history_list[id].tvec - detect_history_list[id + 1].tvec)
                    R_scale = dis / self.observer_params.R_scale_dis_diff_denominator
                    R = np.eye(3) * R_scale
                    tv_vec = (predict_history_list[id].tvec - predict_history_list[id + 1].tvec) / (predict_history_list[id].time - predict_history_list[id + 1].time)
                    X_bias = tv_vec * period_new
                    A = np.eye(3)
                    Z = detect_history_list[id].tvec
                    tvec_kf.predict(A,Z,X_bias,None,R)

                    # predict rvec
                    dis = np.linalg.norm(detect_history_list[id].rvec - detect_history_list[id + 1].rvec)
                    R_scale = dis / self.observer_params.R_scale_dis_diff_denominator
                    R = np.eye(3) * R_scale
                    rv_vec = (predict_history_list[id].rvec - predict_history_list[id + 1].rvec) / (predict_history_list[id].time - predict_history_list[id + 1].time)
                    X_bias = rv_vec * period_new
                    A = np.eye(3)
                    Z = detect_history_list[id].rvec
                    
                    rvec_kf.predict(A,Z,X_bias,None,R)
                    
                    confidence = 0.5
                
                else:
                    # predict tvec
                    dis = np.linalg.norm(detect_history_list[id].tvec - detect_history_list[id + 1].tvec)
                    R_scale = dis / self.observer_params.R_scale_dis_diff_denominator
                    R = np.eye(3) * R_scale
                    tv_vec = (predict_history_list[id].tvec - predict_history_list[id + 1].tvec) / (predict_history_list[id].time - predict_history_list[id + 1].time)                
                    tv_vec_old = (predict_history_list[id + 1].tvec - predict_history_list[id + 2].tvec) / (predict_history_list[id + 1].time - predict_history_list[id + 2].time)                    
                    ta_vec = (tv_vec - tv_vec_old) / (predict_history_list[id].time - predict_history_list[id + 1].time)
                    X_bias = tv_vec * period_new + ta_vec * (period_new ** 2) / 2
                    A = np.eye(3)
                    Z = detect_history_list[id].tvec
                    tvec_kf.predict(A,Z,X_bias,None,R)

                    # predict rvec
                    dis = np.linalg.norm(detect_history_list[id].rvec - detect_history_list[id + 1].rvec)
                    R_scale = dis / self.observer_params.R_scale_dis_diff_denominator
                    R = np.eye(3) * R_scale
                    rv_vec = (predict_history_list[id].rvec - predict_history_list[id + 1].rvec) / (predict_history_list[id].time - predict_history_list[id + 1].time)
                    rv_vec_old = (predict_history_list[id + 1].rvec - predict_history_list[id + 2].rvec) / (predict_history_list[id + 1].time - predict_history_list[id + 2].time)
                    ra_vec = (rv_vec - rv_vec_old) / (predict_history_list[id].time - predict_history_list[id + 1].time)
                    X_bias = rv_vec * period_new + ra_vec * (period_new ** 2) / 2
                    A = np.eye(3)
                    Z = detect_history_list[id].rvec
                    
                    rvec_kf.predict(A,Z,X_bias,None,R)
                    
                    confidence = 0.75

            tvec_predict = tvec_kf.get_cur_state()
            rvec_predict = rvec_kf.get_cur_state()        
        
        return tvec_predict,rvec_predict,cur_time,confidence
          
        
    def __cal_car_rotation_speed(self,
                            armor_name:str)->float:
        
        each_armor_rotation_speed_list = []
        each_armor_confidence_list = []
        
        for each_armor_predict_history_list in self.observer_params.armor_name_to_car_params[armor_name].armor_idx_to_predict_history.values():
            
            tvec_in_car_frame_latest = each_armor_predict_history_list[0].tvec_in_car_frame
            tvec_in_car_frame_old = each_armor_predict_history_list[1].tvec_in_car_frame
            period = each_armor_predict_history_list[0].time - each_armor_predict_history_list[1].time
            
            each_armor_rotation_speed = get_rotation_speed_in_xoz_plane(tvec_in_car_frame_latest, tvec_in_car_frame_old, period)
            each_armor_confidence = each_armor_predict_history_list[0].confidence
            
            each_armor_rotation_speed_list.append(each_armor_rotation_speed)
            each_armor_confidence_list.append(each_armor_confidence)
            
        cur_time = self.observer_params.armor_name_to_car_params[armor_name].armor_idx_to_predict_history[0][0].time
        
        car_rotation_speed = np.average(each_armor_rotation_speed_list,weights=each_armor_confidence_list)
        #car_rotation_speed = each_armor_confidence.index(max(each_armor_confidence))
        
        if self.mode == 'Dbg':
            
            lr1.info(f"Observer: Calculate {armor_name} car rotation_speed {car_rotation_speed} at time {cur_time} with confidence {np.mean(each_armor_confidence_list)}")

        return car_rotation_speed
 
 
    def __find_continous_num(self,
                          armor_history_list:list)->int:
        """Find how many times the armor(id) has been detected continuously latest
        Warning: only detect the continuity of time, not the continuity of position.\n
        Args:
            armor_name (str): _description_
            armor_idx (int): _description_

        Returns:
            int: 1 < continous_num <= history_depth
        """
        
        continous_num = 1
        for i in range(self.observer_params.history_depth):
            dt = armor_history_list[i].time - armor_history_list[i+1].time
            
            if dt > self.observer_params.min_time_between_continous_detection:
                break
            else:
                continous_num += 1
                
        return continous_num
                
    def __check_if_armor_is_focused(self,
                                    armor_name:str,
                                    armor_idx:int
                                    ):
        
        car_params = self.observer_params.armor_name_to_car_params[armor_name]
        armor_latest_params = car_params.armor_idx_to_detect_history[armor_idx][0]
        cur_time = time.time()
        period = cur_time - armor_latest_params.time
        
        if period > self.observer_params.min_time_between_continous_detection:
            return False
        else:
            return True
 
 
    