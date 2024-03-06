import numpy as np
import cv2
from ..os_op.basic import *



def get_other_face_center_pos(tvec_0:np.ndarray,
                                rvec_0:np.ndarray,
                                side_0_length:float,
                                side_1_length:float,
                                armor_nums:int = 4)->list:
    """
    Get the position of the center of the other 3 faces of the cube.
    The 0 face is facing the camera, 1,2,3 is counterclockwise from the 0 face. 
    Returns:
        [tvec_0, tvec_1,...],[rvec_0, rvec_1 ,...]: the position of the center of the other 3 faces of the cube.
    """
    x_unit = np.array([1,0,0])
    y_unit = np.array([0,1,0])
    z_unit = np.array([0,0,1])
    
    rotation_scale_to_y =  np.dot(y_unit,rvec_0) 
    rvec_0_to_y_axis = rotation_scale_to_y * y_unit
    rot_matrix_0 = TRANS_RVEC_TO_ROT_MATRIX(rvec_0_to_y_axis)
    
    
    # x>0 face right
    # y>0 face up
    # z>0 face back
    
    #rotation_scale_to_y > 0 means turn clockwise when thumb is pointing up
    
    if armor_nums == 4:
        
        tvec_1 = tvec_0 + side_0_length/2 * x_unit + side_1_length/2 * y_unit
        tvec_2 = tvec_0 + side_1_length * y_unit
        tvec_3 = tvec_0 - side_0_length/2 * x_unit + side_1_length/2 * y_unit
        
        
        rvec_1 = rvec_0 - z_unit * np.pi/2 
        rvec_2 = rvec_1 - z_unit * np.pi/2 
        rvec_3 = rvec_2 - z_unit * np.pi/2 
        
        #tvec_1 = TRANS_RVEC_TO_ROT_MATRIX(rvec_1) @ tvec_1
        #tvec_2 = TRANS_RVEC_TO_ROT_MATRIX(rvec_2) @ tvec_2
        #tvec_3 = TRANS_RVEC_TO_ROT_MATRIX(rvec_3) @ tvec_3
        
        
        return [tvec_0,tvec_1, tvec_2, tvec_3], [rvec_0, rvec_1, rvec_2, rvec_3]
    
    else:
        tvec_1 = tvec_0 + side_1_length * y_unit
        
        rvec_1 = rvec_0 - y_unit * np.pi + side_1_length * y_unit
        
        #tvec_1 = TRANS_RVEC_TO_ROT_MATRIX(rvec_1) @ tvec_1
        
        
        return [tvec_0, tvec_1], [rvec_0, rvec_1]
    

def get_rotation_speed_in_xoy_plane(tvec_latest:np.ndarray,tvec_old:np.ndarray,dt:float)->float:
    
    """
    Args:
        tvec_latest (np.ndarray): _description_
        tvec_old (np.ndarray): _description_

    Returns:
        float: if > 0, the cube is rotationning counterclockwise in the xoy plane. (x face is right, z face is back)
               if = 0, the cube is not rotationning in the xoy plane or the vectors are too close to zero.
    """
    
    tvec_latest_xoy = tvec_latest.flatten()[[0,1]]
    tvec_old_xoy = tvec_old.flatten()[[0,1]]
    
    if dt == 0:
        return 0
    
    if np.linalg.norm(tvec_latest_xoy) == 0 or np.linalg.norm(tvec_old_xoy) == 0:
        return 0
    
    cos_theta = CAL_COS_THETA(tvec_latest_xoy,tvec_old_xoy)
    
    theta = np.arccos(cos_theta)
    speed = theta/dt
    cross_prod = np.cross(tvec_latest_xoy,tvec_old_xoy)
    
    if cross_prod > 0:
        speed = -speed
        
    return speed


class Kalman_Filter:
    def __init__(self, 
                 Q:np.ndarray,
                 R:np.ndarray, 
                 H:np.ndarray,
                 X_0:Union[np.ndarray,None]=None, 
                 P_0:Union[np.ndarray,None]=None
                 ):
        
        self.Q = Q      # process noise covariance matrix
        self.R = R      # measurement noise covariance matrix
        self.H = H      # measurement matrix
        self.K = None   # Kalman gain matrix
        
        self.X_posterior_predict = X_0    # initial state     
        self.P_posterior_predict = P_0    # initial state covariance matrix
        self.X_prior_predict = None
        self.P_prior_predict = None
        
    def set_initial_state(self, 
                          X_0:np.ndarray, 
                          P_0:np.ndarray):
        self.X_posterior_predict = X_0
        self.P_posterior_predict = P_0
    
    def predict(self, 
                A:np.ndarray,
                Z:np.ndarray,
                X_bias:Union[np.ndarray,None]=None,
                Q_new: Union[np.array,None]=None,
                R_new: Union[np.array,None]=None):
        
        if Q_new is not None:
            self.Q = Q_new
        if R_new is not None:
            self.R = R_new
            
        self._prior_predict(A,X_bias)
        self._correct(Z)
        
        return self.X_posterior_predict
    
    def get_cur_state(self)->np.ndarray:
        return self.X_posterior_predict
    
    def _prior_predict(self,
                       A:np.ndarray,
                       X_bias:Union[np.ndarray,None]=None):
        """
        Predict the state of the system before receiving the measurement.

        Args:
            A (np.ndarray): _description_
        """
        if X_bias is not None:
            self.X_prior_predict = A @ self.X_posterior_predict + X_bias
        else:
            self.X_prior_predict = A @ self.X_posterior_predict
            
        self.P_prior_predict = A @ self.P_posterior_predict @ A.T + self.Q
        lr1.debug(f"KALMAN prior pridict : X_prior_predict:{self.X_prior_predict}, P_prior_predict:{self.P_prior_predict}")
        
    def _correct(self,
                 Z:np.ndarray):
        
        inv_mat = np.linalg.pinv(self.H @ self.P_prior_predict @ self.H.T + self.R)
        
        self.K = self.P_prior_predict @ self.H.T @ inv_mat
        
        self.X_posterior_predict = self.X_prior_predict + self.K @ (Z - self.H @ self.X_prior_predict)
        self.P_posterior_predict = self.X_posterior_predict - self.K @ self.H@ self.P_prior_predict
        
        lr1.debug(f"KALMAN : K:{self.K}, X_posterior_predict:{self.X_posterior_predict}, P_posterior_predict:{self.P_posterior_predict}")

        

def trans_t_to_unix_time(minute:int,
                         second:int,
                         second_frac:float,
                         zero_unix_time:float)->float:
    """
    Transform the time in the format of (minute, second, second_frac) to the unix time.
    Args:
        minute (int): _description_
        second (int): _description_
        second_frac (float): _description_
        zero_unix_time (float): _description_

    Returns:
        float: the unix time.
    """
    return zero_unix_time + minute*60 + second + second_frac/1000000000.0

