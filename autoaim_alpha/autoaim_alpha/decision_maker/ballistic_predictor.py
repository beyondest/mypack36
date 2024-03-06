from ..os_op.basic import *
from ..os_op.decorator import *
from ..os_op.global_logger import *
from ..img.tools import Plt_Dynamic_Window


"""
Air drag coefficient: 
>>> C = 0.47
>>> p = 1.169
>>> import numpy as np
>>> S = np.pi * 16.8**2 * 1e-6 /4
>>> S
0.0002216707776372958
>>> C * p * S
0.00012179257535725942
>>> C * p * S * 30 /2
0.0018268886303588914
"""

class Ballistic_Predictor_Params(Params):
    def __init__(self) -> None:
        super().__init__()
        
        # see ./observer_coordinates.png
        self.camera_x_len = 0.12
        self.camera_y_len = 0.75
        self.muzzle_x_len = 0.15
        self.muzzle_y_len = 0.0
        
        # due to the limitation of the electrical system, the yaw and pitch can only be within certain range
        self.yaw_range = [0,2*np.pi]
        self.pitch_range = [-np.pi/4 , np.pi/4]
        self.bullet_speed = 27 # m/s
        
        # input: initial_velocity_vector, initial_position_vector, flight_time
        # output: final_position_vector
        
        self.g = 9.7985 # m/s^2, Qingdao gravity constant
        self.bullet_mass = 3.2e-3 # kg
        self.k   = 1e-4 # air drag coefficient, dimensionless
        
        self.R_K4_dt_near = 0.001 # time step for Runge-Kutta method, s, if Z < 2
        self.R_K4_dt_far = 0.01
        self.R_K4_error_tolerance = 0.01 # error tolerance for Runge-Kutta method, m
        
        self.newton_max_iter = 5 # max iteration for newton method
        self.newton_error_tolerance = 0.01 # error tolerance for newton method, radian angle
        self.newton_dx = 0.001 # step for newton method, radian angle
        
        self.target_min_hei_above_ground = 0.10 # m, min height of target above ground
        self.max_shooting_dis_in_pivot_frame = 10 # m, max shooting distance in camera frame, consider target min height
        self.max_shooting_hei_above_ground = 0    # need to be updated by cal_max_shooting_hei_above_ground()
        
        self.gun_pivot_height_above_ground = 0.375 # m, height of camera above ground
        self.learning_rate = 0.002 # learning rate for gradient descent method
        self.gradient_error_tolerance = 0.001 # error tolerance for gradient descent method, radian angle
        
        
        self.camera_init_theta = None
        self.muzzle_init_theta = None
        self.camera_radius = None
        self.muzzle_radius = None
        self.camera_pos_in_gun_pivot_frame = None
        self.muzzle_pos_in_gun_pivot_frame = None
        
class Ballistic_Predictor:
    def __init__(self,
                 mode:str = 'Dbg',
                 params_yaml_path:Union[str,None] = None,
                 if_show_ballistic_trajectory:bool = False):
        
        CHECK_INPUT_VALID(mode, 'Dbg','Rel')
        self.mode = mode
        self.params = Ballistic_Predictor_Params()
        self.if_show_ballistic_trajectory = if_show_ballistic_trajectory
        
        if params_yaml_path is not None:
            self.params.load_params_from_yaml(params_yaml_path)
            
        if self.if_show_ballistic_trajectory:
            self.plt_dynamic_window = Plt_Dynamic_Window()
        
        self.params.camera_init_theta = np.arctan2(self.params.camera_y_len, self.params.camera_x_len)
        self.params.muzzle_init_theta = np.arctan2(self.params.muzzle_y_len, self.params.muzzle_x_len)
        self.params.camera_radius = np.sqrt(self.params.camera_x_len**2 + self.params.camera_y_len**2)
        self.params.muzzle_radius = np.sqrt(self.params.muzzle_x_len**2 + self.params.muzzle_y_len**2)
        
        self.params.camera_pos_in_gun_pivot_frame = np.array([0.0, 
                                                              self.params.camera_radius * np.cos(self.params.camera_init_theta),
                                                              self.params.camera_radius * np.sin(self.params.camera_init_theta)])
        self.params.muzzle_pos_in_gun_pivot_frame = np.array([0.0, 
                                                              self.params.muzzle_radius * np.cos(self.params.muzzle_init_theta),
                                                              self.params.muzzle_radius * np.sin(self.params.muzzle_init_theta), ])

        
     
        
    def save_params_to_yaml(self,yaml_path:str):
        self.params.camera_init_theta = None
        self.params.muzzle_init_theta = None
        self.params.camera_radius = None
        self.params.muzzle_radius = None
        self.params.camera_pos_in_gun_pivot_frame = None
        self.params.muzzle_pos_in_gun_pivot_frame = None
        self.params.save_params_to_yaml(yaml_path)
        
            
    def get_fire_yaw_pitch( self, 
                            target_pos_in_camera_frame:np.ndarray,
                            cur_yaw:float, 
                            cur_pitch:float)->list:
        """
        Input:
            target_pos: the position of the target 
            cur_yaw: the current yaw from electrical system (radian angle)
            cur_pitch: the current pitch from electrical system (radian angle)
            
        Returns:
            list of dicts:
                yaw: absolute yaw angle from electrical system (radian angle) (0-2pi)
                pitch: absolute pitch angle from electrical system (radian angle) (-pi/3 to pi/3)
                bullet flight time: _description_
                if_success: If the bullet can reach the target return True, otherwise return False.
        """
        target_hei_above_ground = target_pos_in_camera_frame[2] + self.params.camera_radius * np.sin(cur_pitch) + self.params.gun_pivot_height_above_ground
        
        
        if target_pos_in_camera_frame[1] > self.params.max_shooting_dis_in_pivot_frame \
            or target_hei_above_ground < self.params.target_min_hei_above_ground\
                or target_hei_above_ground > self.params.max_shooting_hei_above_ground:
                
            if self.mode == 'Dbg':
                
                lr1.warn(f"Ballistic_Predictor : Target out of range, return current yaw and pitch,target_hei_above_ground: {target_hei_above_ground:.3f}, target_dis_in_cam_frame: {target_pos_in_camera_frame[1]:.3f}, shooting_hei_range: {self.params.target_min_hei_above_ground:.3f} - {self.params.max_shooting_hei_above_ground:.3f}, max_shooting_dis_in_pivot_frame: {self.params.max_shooting_dis_in_pivot_frame:.3f}")
                
            return [cur_yaw, cur_pitch, 0, False]
        
        self._update_camera_pos_in_gun_pivot_frame(cur_yaw, cur_pitch)
        target_pos_in_gun_pivot_frame = target_pos_in_camera_frame + self.params.camera_pos_in_gun_pivot_frame
        tvec_xoy = target_pos_in_gun_pivot_frame[[0,1]]
        tvec_yoz = target_pos_in_gun_pivot_frame[[1,2]]
        
        cos_theta_with_z_axis = CAL_COS_THETA(tvec_xoy, np.array([0,1]))
        if tvec_xoy[0] > 0:
            required_yaw = -np.arccos(cos_theta_with_z_axis) 
        else:
            required_yaw = np.arccos(cos_theta_with_z_axis)
            
        [required_pitch , flight_time, if_success] , solve_time = self._cal_pitch_by_newton(tvec_yoz)
        
        required_pitch = required_pitch
        required_yaw = required_yaw + cur_yaw
        if required_yaw < 0:
            required_yaw = required_yaw + 2*np.pi
        if required_yaw > 2*np.pi:
            required_yaw = required_yaw - 2*np.pi
        
        if not if_success:
            required_pitch = cur_pitch
            required_yaw = cur_yaw
            
        if self.mode == 'Dbg':
            lr1.debug(f"Ballistic_Predictor : required_yaw: {required_yaw} , required_pitch: {required_pitch} ,bullet_flight_time: {flight_time} ,if_success: {if_success} , solve_time: {solve_time}")
            
            
        return required_yaw, required_pitch, flight_time, if_success
    
    
    def cal_max_shooting_dis_by_gradient(self)->float:
        '''
        Returns:
            float: max shooting distance in camera frame, consider target min height
        '''
        target_hei_in_world_frame = self.params.target_min_hei_above_ground
        if target_hei_in_world_frame > self.params.max_shooting_hei_above_ground:
            lr1.error(f"Ballistic_Predictor : target min height > max shooting height, {target_hei_in_world_frame} > {self.params.max_shooting_hei_above_ground}")
            lr1.error(f"Ballistic_Predictor : cal max shooting dis by gradient failed, return 0.0")
            return 0.0
        
        target_hei_in_pivot_frame = target_hei_in_world_frame - self.params.gun_pivot_height_above_ground
        
        pitch_start = self.params.pitch_range[0]
        pitch_end = self.params.pitch_range[1]
        
        x = pitch_start
        x_new = (pitch_start + pitch_end) / 2
        dx = self.params.newton_dx
        target_tvec_yoz_in_pivot_frame = np.array([0.0, target_hei_in_pivot_frame])
        
        while True:
            [dis_diff , _] , _ = self._R_K4_air_drag_ballistic_model(x_new, target_tvec_yoz_in_pivot_frame,'hei')
            actual_dis = 0 + dis_diff
            [dis_diff_ , _] , _ = self._R_K4_air_drag_ballistic_model(x_new + dx, target_tvec_yoz_in_pivot_frame,'hei')
            dfx = (dis_diff_ - dis_diff) / dx
            #lr1.info(f'Ballistic_Predictor : dfx: {dfx} , x_new: {x_new} , x: {x}')
            x = x_new
            x_new = x + dfx * self.params.learning_rate
            if dfx < self.params.gradient_error_tolerance:
                break
        
        actual_dis = float(actual_dis)
        self.params.max_shooting_dis_in_pivot_frame = actual_dis
        lr1.info(f"Ballistic_Predictor : target_min_hei_above_ground: {self.params.target_min_hei_above_ground:.3f} , max_shooting_dis_in_pivot_frame: {actual_dis:.3f}")
                
        return actual_dis
            
            
    def cal_max_shooting_hei_above_ground(self,max_pitch)->tuple:
        """_summary_

        Args:
            max_pitch (_type_): _description_

        Returns:
            tuple: 
                - max_shooting_hei_above_ground: _description_
                - bullet_drop_on_ground_time: _description_
        """
        
        bullet_drop_on_ground_tvec = np.array([0.0, -self.params.gun_pivot_height_above_ground])
        [max_height_above_ground, bullet_drop_on_ground_time], _ = self._R_K4_air_drag_ballistic_model(max_pitch,
                                                                                                       bullet_drop_on_ground_tvec,
                                                                                                       'cal_hei')
        self.params.max_shooting_hei_above_ground = float(max_height_above_ground)
        lr1.info(f"Ballistic_Predictor : max_shooting_hei_above_ground: {self.params.max_shooting_hei_above_ground:.3f}, bullet_drop_on_ground_time: {bullet_drop_on_ground_time:.3f}")
        lr1.info(f"Ballistic_Predictor : k : {self.params.k} ")
        
        return self.params.max_shooting_hei_above_ground, bullet_drop_on_ground_time
        
    
    def _update_camera_pos_in_gun_pivot_frame(self, 
                                              cur_yaw:float, 
                                              cur_pitch:float)->np.ndarray:
        """_summary_

        Args:
            cur_yaw (float): current yaw from electrical system (radian angle)
            cur_pitch (float): current pitch from electrical system (radian angle)

        Returns:
            np.ndarray: _description_
        """
        
        camera_theta = cur_pitch + self.params.camera_init_theta 
        muzzle_theta = cur_pitch + self.params.muzzle_init_theta 
        
        self.params.camera_pos_in_gun_pivot_frame = np.array([0.0, self.params.camera_radius * np.sin(camera_theta) ,self.params.camera_radius * np.cos(camera_theta)])
        self.params.muzzle_pos_in_gun_pivot_frame = np.array([0.0, self.params.muzzle_radius * np.sin(muzzle_theta) ,self.params.muzzle_radius * np.cos(muzzle_theta)])
   
    @timing(1)
    def _R_K4_air_drag_ballistic_model( self,
                                        pitch:float,
                                        tvec_yoz_in_pivot_frame:np.ndarray,
                                        specified_hei_or_dis:str='dis'
                                        )->list:
        """@timing(1)\n
        A ballistic model considering air drag and gravity.
        f_air = -k_air_drag * |v| ** 2 * v / |v|
        f_gravity = -g * y_unit

        Using the Runge-Kutta method to solve.
        
        Notice:
            specified_hei_or_dis: 'hei' or 'dis' or 'cal_hei
        Returns:
            [x , flight_time:float]
            if specified_hei_or_dis == 'hei':
                x = dis_diff
            elif specified_hei_or_dis == 'dis':
                x = hei_diff
            elif specified_hei_or_dis == 'cal_hei':
                x = max_height
                t = bullet_drop_on_ground_time
            
        """
        
        actual_angle = self.params.muzzle_init_theta + pitch
        tvec_init = np.array([np.cos(actual_angle) * self.params.muzzle_radius,
                              np.sin(actual_angle) * self.params.muzzle_radius])
        v_init = np.array([np.cos(pitch) * self.params.bullet_speed, 
                           np.sin(pitch) * self.params.bullet_speed])
        
        v = v_init
        r = tvec_init
        k = self.params.k
        g = self.params.g
        m = self.params.bullet_mass
        dt = self.params.R_K4_dt_near if tvec_yoz_in_pivot_frame[0] < 1.5 else self.params.R_K4_dt_far 
        t = 0.0
        error = 100
        
        if self.if_show_ballistic_trajectory:
            trajectory = []
        if specified_hei_or_dis == 'cal_hei':
            rt = r
            return_var = 1
            
        else:
            rt = tvec_yoz_in_pivot_frame
            return_var = 0
            if specified_hei_or_dis == 'hei':
                init_diff = abs(tvec_init[1] - tvec_yoz_in_pivot_frame[1])
            else:
                init_diff = abs(tvec_init[0] - tvec_yoz_in_pivot_frame[0])
            
            
        error_thresh = self.params.R_K4_error_tolerance if specified_hei_or_dis != 'cal_hei' else 1e-5 
        while error > error_thresh:
            if specified_hei_or_dis == 'cal_hei':
                rt = r
            
            v_l = np.linalg.norm(v)
            
            v_u = v / v_l
            f = -k * v_l**2 * v_u
            a = np.array([0.0, -g]) + f / m
            
            k1 = dt * a
            k2 = dt * (a + 0.5 * k1)
            k3 = dt * (a + 0.5 * k2)
            k4 = dt * (a + k3)
            
            v = v + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            r = r + v * dt
            
            t = t + dt
            
            hei_diff = r[1] - rt[1]
            dis_diff = r[0] - rt[0]
            

            
            if self.if_show_ballistic_trajectory:
                trajectory.append(r.copy())
            if specified_hei_or_dis == 'hei':
                error = abs(hei_diff)
                if abs(r[1] - tvec_init[1]) > init_diff + self.params.R_K4_error_tolerance:
                    break
                    
            elif specified_hei_or_dis == 'dis':
                error = abs(dis_diff)
                if abs(r[0] - tvec_init[0]) > init_diff + self.params.R_K4_error_tolerance:
                    break
                
            elif specified_hei_or_dis == 'cal_hei':
                error = abs(r[1] - rt[1])
                if error <= error_thresh:
                    max_height = float(r[1]) + self.params.gun_pivot_height_above_ground
                    error = 100
                    error_thresh = self.params.R_K4_error_tolerance
                    specified_hei_or_dis = 'hei'
                    rt = tvec_yoz_in_pivot_frame
            
                

        if self.if_show_ballistic_trajectory:
            trajectory = np.array(trajectory)
            x = trajectory[:,0]
            y = trajectory[:,1]
            self.plt_dynamic_window.update(x,y)
            
        if specified_hei_or_dis == 'dis':
            actual_diff = hei_diff
        elif specified_hei_or_dis == 'hei':
            actual_diff = dis_diff
        
        return [actual_diff, t] if return_var == 0 else [max_height, t]
        
    
    @timing(1)
    def _cal_pitch_by_newton(self,
                             target_tvec_yoz_in_pivot_frame:np.ndarray)->list:
        """@timing(1)\n
        Only Success when target is in range and pitch is in range.\n
        Args:
            target_tvec_yoz_in_pivot_frame (np.ndarray): _description_

        Returns:
            list: [pitch, flight_time:float, if_success:bool]
            
        """
        c = 1.0
        a = self.params.pitch_range[0]
        b = self.params.pitch_range[1]
        
        
        RK4_spend_time_all = 0.0
        count = 0
        
        x = a
        x_new = (a + b) / 2
        dx = self.params.newton_dx
        if_success = False
        while True:
            
            [fx , flight_time] , RK4_spend_time = self._R_K4_air_drag_ballistic_model(x_new, target_tvec_yoz_in_pivot_frame,'dis')
            [fx_ , _] , RK4_spend_time2 = self._R_K4_air_drag_ballistic_model(x_new + dx, target_tvec_yoz_in_pivot_frame,'dis')
            dfx = (fx_ - fx) / dx 
            
            x = x_new
            if dfx == 0:
                x_new = x
                break
            else:
                x_new = x - fx / dfx * c
            
            RK4_spend_time_all += RK4_spend_time + RK4_spend_time2
                
            count += 1    
            
            
            if abs(fx) < self.params.newton_error_tolerance:
                if INRANGE(x, self.params.pitch_range):
                    if_success = True
                else:
                    lr1.warn(f"Ballistic_Predictor : newton failed, solved pitch out of range, {x}")
                break
            if count >= self.params.newton_max_iter:
                if abs(fx) < self.params.newton_error_tolerance:
                    if INRANGE(x, self.params.pitch_range):
                        if_success = True
                    else:
                        lr1.warn(f"Ballistic_Predictor : newton failed, solved pitch out of range, {x}")
                    break
                else:
                    if 1/c<3:
                        
                        c = c / 2
                        count = 0
                    else:
                        lr1.warn(f"Ballistic_Predictor : newton failed, iter too many times, final error {fx}")
                        break
                
            
                    
                
            
            #print(f"x_new: {x_new} , x: {x} , fx: {fx} , dfx: {dfx}")
            
        
        if self.mode == 'Dbg':
            lr1.debug(f"Ballistic_Predictor : RK4_spend_time_all: {RK4_spend_time_all}, average: {RK4_spend_time_all / count / 2}")
            lr1.debug(f"Ballistic_Predictor : newton iterations: {count}")
        
        return [x_new, flight_time, if_success]
            
        
    

            
            