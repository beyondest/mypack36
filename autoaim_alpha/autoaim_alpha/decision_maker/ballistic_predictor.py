from ..os_op.basic import *
from ..os_op.decorator import *
from ..os_op.global_logger import *





class Ballistic_Predictor_Params(Params):
    def __init__(self) -> None:
        super().__init__()
        
        # see ./observer_coordinates.png
        self.camera_x_len = 0.0
        self.camera_y_len = 0.0
        self.muzzle_x_len = 0.0
        self.muzzle_y_len = 0.0
        
        self.camera_init_theta = np.arctan2(self.camera_y_len, self.camera_x_len)
        self.muzzle_init_theta = np.arctan2(self.muzzle_y_len, self.muzzle_x_len)
        self.camera_radius = np.sqrt(self.camera_x_len**2 + self.camera_y_len**2)
        self.muzzle_radius = np.sqrt(self.muzzle_x_len**2 + self.muzzle_y_len**2)
        
        self.camera_pos_in_gun_pivot_frame = np.array([0.0, self.camera_radius * np.sin(self.camera_init_theta) , np.cos(self.camera_init_theta)])
        self.muzzle_pos_in_gun_pivot_frame = np.array([0.0, self.muzzle_radius * np.sin(self.muzzle_init_theta), np.cos(self.muzzle_init_theta)])
        
        
       
        
        # due to the limitation of the electrical system, the yaw and pitch can only be within certain range
        self.yaw_range = [0,2*np.pi]
        self.pitch_range = [-np.pi/3 , np.pi/3]
        
        
        self.bullet_speed = 10 # m/s
        
        # input: initial_velocity_vector, initial_position_vector, flight_time
        # output: final_position_vector
        
        self.g = 9.81 # m/s^2
        self.bullet_mass = 0.1 # kg
        self.k   = 0.01 # air drag coefficient, dimensionless
        
        
        self.R_K4_dt = 0.01 # time step for Runge-Kutta method, s
        self.R_K4_error_tolerance = 1e-3 # error tolerance for Runge-Kutta method, m
        
        self.newton_max_iter = 100 # max iteration for newton method
        self.newton_error_tolerance = 1e-3 # error tolerance for newton method, radian angle
        self.newton_dx = 0.01 # step for newton method, radian angle
        
        self.target_min_hei_above_ground = 0.1 # m, min height of target above ground
        self.max_shooting_dis_in_camera_frame = 10 # m, max shooting distance in camera frame, consider target min height
        self.gun_pivot_height_above_ground = 0.5 # m, height of camera above ground
        self.learning_rate = 0.01 # learning rate for gradient descent method
        self.gradient_error_tolerance = 1e-3 # error tolerance for gradient descent method, radian angle
        
        
class Ballistic_Predictor:
    
    
    def __init__(self,
                 mode:str = 'Dbg',
                 params_yaml_path:Union[str,None] = None,
                 if_cal_max_shooting_dis_by_gradient:bool = False,
                 if_show_ballistic_trajectory:bool = False):
        
        CHECK_INPUT_VALID(mode, ['Dbg', 'Rel'])
        self.mode = mode
        self.params = Ballistic_Predictor_Params()
        self.if_show_ballistic_trajectory = if_show_ballistic_trajectory
        if params_yaml_path is not None:
            self.params.load_params_from_yaml(params_yaml_path)
        
        
        if if_cal_max_shooting_dis_by_gradient:
            
            max_dis ,t = self._cal_max_shooting_dis_by_gradient(self.params.target_min_hei_above_ground)
            self.params.max_shooting_dis_in_camera_frame = max_dis
            if self.mode == 'Dbg':
                lr1.info(f"Decision_maker : cal_time {t}")
                lr1.info(f"Decision_maker : target_min_hei_above_ground: {self.params.target_min_hei_above_ground} , max_shooting_dis_in_camera_frame: {max_dis}")
    
    def save_params_to_yaml(self,yaml_path:str):
        self.params.save_params_to_yaml(yaml_path)
        
            
    def get_fire_yaw_pitch(self, 
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
                - yaw: absolute yaw angle from electrical system (radian angle) (0-2pi)
                - pitch: absolute pitch angle from electrical system (radian angle) (-pi/3 to pi/3)
                - bullet flight time: _description_
                - if_success: If the bullet can reach the target return True, otherwise return False.
        """
        target_hei_above_ground = target_pos_in_camera_frame[1] + self.params.camera_y_len + self.params.gun_pivot_height_above_ground
        
        if target_pos_in_camera_frame[2] > self.params.max_shooting_dis_in_camera_frame or target_pos_in_camera_frame[1] \
            or target_hei_above_ground < self.params.target_min_hei_above_ground:
                
            if self.mode == 'Dbg':
                
                lr1.info(f"Decision_maker : target_dis: {target_pos_in_camera_frame[2]} , target_hei_above_ground: {target_hei_above_ground} ,\
                    \nmax_shooting_dis_in_camera_frame: {self.params.max_shooting_dis_in_camera_frame} , target_min_hei_above_ground: {self.params.target_min_hei_above_ground}")
                lr1.info(f"Decision_maker : Target out of range, return current yaw and pitch")
                
            return [cur_yaw, cur_pitch, 0, False]
        
        self._update_camera_pos_in_gun_pivot_frame(cur_yaw, cur_pitch)
        
        target_pos_in_gun_pivot_frame = target_pos_in_camera_frame + self.params.camera_pos_in_gun_pivot_frame
        tvec_xoz = target_pos_in_gun_pivot_frame[[0,2]]
        tvec_zoy = target_pos_in_gun_pivot_frame[[2,1]]
        
        cos_theta_with_z_axis = CAL_COS_THETA(tvec_xoz, np.array([0,1]))
        if tvec_xoz[0] > 0:
            required_yaw = np.arccos(cos_theta_with_z_axis) 
        else:
            required_yaw = -np.arccos(cos_theta_with_z_axis)
        
        [required_pitch , flight_time, if_success] , solve_time = self._cal_pitch_by_newton(tvec_zoy)
        
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
            lr1.info(f"Decision_maker : required_yaw: {required_yaw} , required_pitch: {required_pitch} ,bullet_flight_time: {flight_time} ,\
                    \nif_success: {if_success} , solve_time: {solve_time}")
            
        return required_yaw, required_pitch, flight_time, if_success
    
    
    
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
        self.params.camera_pos_in_gun_pivot_frame = np.array([0.0, self.params.camera_radius * np.sin(camera_theta) , np.cos(camera_theta)])
        self.params.muzzle_pos_in_gun_pivot_frame = np.array([0.0, self.params.muzzle_radius * np.sin(muzzle_theta) , np.cos(muzzle_theta)])
    
    @timing(1)
    def _cal_max_shooting_dis_by_gradient(self, target_hei_in_world_frame:float)->float:
        '''
        @timing(1)
        Returns:
            float: max shooting distance in camera frame, consider target min height
        '''
        target_hei_in_camera_frame = target_hei_in_world_frame - self.params.gun_pivot_height_above_ground - self.params.camera_y_len
        
        pitch_start = self.params.pitch_range[0]
        pitch_end = self.params.pitch_range[1]
        
        x = pitch_start
        x_new = (pitch_start + pitch_end) / 2
        dx = self.params.newton_dx
        target_tvec_zoy = np.array([0.0, target_hei_in_camera_frame])
        
        while True:
            [dis_diff , _] , _ = self._R_K4_air_drag_ballistic_model(x_new, target_tvec_zoy,'hei')
            actual_dis = 0 + dis_diff
            [dis_diff_ , _] , _ = self._R_K4_air_drag_ballistic_model(x_new + dx, target_tvec_zoy,'hei')
            dfx = (dis_diff_ - dis_diff) / dx
            x = x_new
            x_new = x + dfx * self.params.learning_rate
            if dfx < self.params.gradient_error_tolerance:
                break
        
        
        return actual_dis
            
            
        
        
        
    
    @timing(1)
    def _R_K4_air_drag_ballistic_model(  self,
                                        pitch:float,
                                        tvec_zoy_final:np.ndarray,
                                        specified_hei_or_dis:str='dis'
                                        )->list:
        """@timing(1)
        A ballistic model considering air drag and gravity.
        f_air = -k_air_drag * v
        f_gravity = -g * y_unit

        Using the Runge-Kutta method to solve.
        
        Notice:
            specified_hei_or_dis: 'hei' or 'dis'
        Returns:
            [actual_diff , flight_time:float]
            actual_diff = hei_diff if specified_hei_or_dis == 'dis' else dis_diff
        """
        actual_angle = self.params.muzzle_init_theta + pitch
        tvec_init = np.array([np.cos(actual_angle) * self.params.muzzle_radius, np.sin(actual_angle) * self.params.muzzle_radius])
        v_init = np.array([np.cos(pitch) * self.params.bullet_speed, np.sin(pitch) * self.params.bullet_speed])
        
        v = v_init
        r = tvec_init
        rt = tvec_zoy_final
        k = self.params.k
        g = self.params.g
        m = self.params.bullet_mass
        dt = self.params.R_K4_dt
        t = 0.0
        error = 100
        
        while error > self.params.R_K4_error_tolerance:
            f = -k * v
            
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
            
            if specified_hei_or_dis == 'hei':
                error = abs(hei_diff)
            else:
                error = abs(dis_diff)
        
        
        actual_diff = hei_diff if specified_hei_or_dis == 'dis' else dis_diff
        return [actual_diff, t]
        
    
    
    @timing(1)
    def _cal_pitch_by_newton(self,
                             target_tvec_zoy:np.ndarray)->list:
        """@timing(1)\n
        Only Success when target is in range and pitch is in range.\n
        Args:
            target_tvec_zoy (np.ndarray): _description_

        Returns:
            list: [pitch, flight_time:float, if_success:bool]
        """
        a = self.params.pitch_range[0]
        b = self.params.pitch_range[1]
        
        if self.mode == 'Dbg':
            RK4_spend_time_all = 0.0
            count = 0
        
        x = a
        x_new = (a + b) / 2
        dx = self.params.newton_dx
        if_success = False
        while True:
            
            [fx , flight_time] , RK4_spend_time = self._R_K4_air_drag_ballistic_model(x_new, target_tvec_zoy,'dis')
            [fx_ , _] , RK4_spend_time2 = self._R_K4_air_drag_ballistic_model(x_new + dx, target_tvec_zoy,'dis')
            dfx = (fx_ - fx) / dx
            x = x_new
            x_new = x - fx / dfx
            
            if self.mode == 'Dbg':
                RK4_spend_time_all += RK4_spend_time + RK4_spend_time2
                count += 1    
                
            if abs(x_new - x) < self.params.newton_error_tolerance:
                if INRANGE(x, self.params.pitch_range):
                    if_success = True
                break
            if count >= self.params.newton_max_iter:
                break
            
           
            
        
        
        if self.mode == 'Dbg':
            lr1.info(f"Decision_maker : RK4_spend_time_all: {RK4_spend_time_all}, average: {RK4_spend_time_all / count / 2}")
            lr1.info(f"Decision_maker : newton iterations: {count}")
        
        
        
        return [x_new, flight_time, if_success]
            
        


            
            