
from ..os_op.basic import *
from .const import *
import numpy as np
import cv2


class PNP_Params(Params):
    def __init__(self):
        super().__init__()
            
        self.img_shrink_scale = IMG_SHRINK_SCALE
        self.small_obj_points = SMALL_ARMOR_REC_POINTS
        self.big_obj_points = BIG_ARMOR_REC_POINTS

        self.mtx = MV1_MTX
        self.dist = MV1_DIST
        self.obj_wid_in_world = 100.0 # unit: mm
        self.obj_hei_in_world = 100.0 # unit: mm
        self.method_name = 'pnp'
        self.small_armor_name_list = ['2x','3x','4x','5x','basex','sentry',
                                      'B2','B3','B4','B5',
                                      'R2','R3','R4','R5'
                                      ]
        
        self.expand_rate = 1.5 # expand light bar to get the whole armor







################################################Depth Estimator################################################3

class Depth_Estimator:
    def __init__(self,
                 depth_estimator_config_yaml :Union[str,None],
                 mode:str = 'Dbg'
                 ):
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        
        self.mode = mode
        
        self.pnp_params = PNP_Params()
        if depth_estimator_config_yaml is not None:
            self.pnp_params.load_params_from_yaml(depth_estimator_config_yaml)
        else:
            lr1.warning(f'Depth_Estimator: depth_estimator_config_yaml is None, using default parameters')
                

        
        self.if_enable_trackbar_config = False
        
        
    def get_result(self,input):
        """

        Returns:
            x,y,z: the position of the object in the camera coordinate system.
            
        Method:
            pnp: Perspective-n-Point algorithm.
                input: 
                    big_rec, obj_class = 'big' or'small': the coordinates of the object in the image.
                output: 
                    [x,y,z],rvecs: tvec and rvec in camera frame, unit: m.
                
            pnp_fix: PnP algorithm with fixed parameters.
                input: 
                    big_rec, obj_class = 'big' or'small': the coordinates of the object in the image and the class of the object.
                output: 
                    [x,y,z],rvecs: tvec and rvec in camera frame, unit: m.
                
        """
        
        if self.pnp_params.method_name  == "pnp":
            big_rec = input[0]
            obj_wid_in_world = self.pnp_params.obj_wid_in_world
            obj_hei_in_world = self.pnp_params.obj_hei_in_world
            
            # up left / up right / down right / down left
            obj_pts = np.array([[-obj_wid_in_world/2, -obj_hei_in_world/2, 0],
                                [obj_wid_in_world/2, -obj_hei_in_world/2, 0],
                                [obj_wid_in_world/2, obj_hei_in_world/2, 0],
                                [-obj_wid_in_world/2, obj_hei_in_world/2, 0]], dtype=np.double)
            
            return self._PNP(big_rec,obj_pts)
        
        elif self.pnp_params.method_name == 'pnp_fix':
            big_rec = input[0]
            obj_class = input[1]
            if obj_class == 'big':
                obj_pts = np.array(self.pnp_params.big_obj_points,dtype=np.double)
            elif obj_class =='small':
                obj_pts = np.array(self.pnp_params.small_obj_points,dtype=np.double)
            else:
                lr1.critical(f'Invalid obj_class: {obj_class}')
                raise ValueError(f'Invalid obj_class: {obj_class}')
            
            return self._PNP(big_rec,obj_pts)
        
    def save_params_to_yaml(self,yaml_path:str):
        if not os.path.exists(yaml_path):
            os.makedirs(yaml_path)
        if  self.method == 'pnp_fix' or self.method == 'pnp':
            self.pnp_params.save_params_to_yaml(os.path.join(yaml_path,'pnp_params.yaml'))
        
    def enable_trackbar_config(self,
                               window_name:str = 'depth_estimator_config',
                               save_params_key:str = 'n',
                               save_params_yaml_path:str = './tmp_depth_estimator_config.yaml'):
        self.if_enable_trackbar_config = True
        self.config_window_name =window_name
        self.save_params_key = save_params_key
        self.save_params_yaml_path = save_params_yaml_path
        def for_trackbar(x):
            pass
        cv2.namedWindow(window_name,cv2.WINDOW_FREERATIO)
        cv2.createTrackbar('obj_wid_in_world_mm',window_name,10,1000,for_trackbar)
        cv2.createTrackbar('obj_hei_in_world_mm',window_name,10,1000,for_trackbar)
        
        cv2.setTrackbarPos('obj_wid_in_world_mm',window_name,int(self.pnp_params.obj_wid_in_world))
        cv2.setTrackbarPos('obj_hei_in_world_mm',window_name,int(self.pnp_params.obj_hei_in_world))
        
            
    
    def __detect_trackbar_config(self):
        
        self.pnp_params.obj_wid_in_world = float(cv2.getTrackbarPos('obj_wid_in_world_mm',self.config_window_name))
        self.pnp_params.obj_hei_in_world = float(cv2.getTrackbarPos('obj_hei_in_world_mm',self.config_window_name))
        if cv2.waitKey(1) == ord(self.save_params_key):
            self.pnp_params.save_params_to_yaml(self.save_params_yaml_path)
        

    
    def _PNP(self,big_rec,obj_pts:np.ndarray):
        if big_rec is None:
            lr1.warning(f'big_rec is None')
            return None
        
        if self.if_enable_trackbar_config:
            self.__detect_trackbar_config()
            
        img_points = np.array(big_rec, dtype=np.double)
        img_points = img_points / self.pnp_params.img_shrink_scale
        
        
        success, rvecs, tvecs = cv2.solvePnP(obj_pts, img_points,
                                            np.array(self.pnp_params.mtx, dtype=np.double),
                                            np.array(self.pnp_params.dist, dtype=np.double))
        if not success:
            lr1.warning(f'PnP Failed, big_rec: {big_rec}, obj_pts: {obj_pts}')
            return None
        
        tvecs = np.array(tvecs)
        x = tvecs[0][0] / 1000.0  # unit: m
        y = tvecs[1][0] / 1000.0  # unit: m
        z = tvecs[2][0] / 1000.0  # unit: m
        
        y,z = z,y # change the order of y and z, cause y is deep in rviz2
        z = -z # when target is up, z add, when target is down, z minus.
        
        rx,ry,rz = rvecs[0][0],rvecs[1][0],rvecs[2][0]
        ry, rz = rz, ry # change the order of ry and rz
        
        if self.mode == 'Dbg':
            lr1.debug(f"big_rec: {big_rec}, img_points: {img_points}, obj_pts: {obj_pts},")
            lr1.debug(f'PnP result: x: {x:.4f}, y: {y:.4f}, z: {z:.4f}, rx: {rx:.4f}, ry: {ry:.4f}, rz: {rz:.4f}')
        
        return [x,y,z],[rx,ry,rz]
        
    def _reverse_PNP(self,
                     point_to_camera_coordinate:list,
                     mtx:list,
                     dist:list):
        '''
        point_to_camera_coordinate: np.array([[x,y,z]])
        mtx: [[fx,0,cx],
              [0,fy,cy],
              [0,0,1]]
        dist: [[k1,k2,p1,p2,k3]]
        return: [u,v]
        '''
        u = mtx[0][0] * point_to_camera_coordinate[0] / point_to_camera_coordinate[2] + mtx[0][2]
        v = mtx[1][1] * point_to_camera_coordinate[1] / point_to_camera_coordinate[2] + mtx[1][2]
        r = np.sqrt(u ** 2 + v ** 2)
        
        
        k1 = dist[0][0]
        k2 = dist[0][1]
        p1 = dist[0][2]
        p2 = dist[0][3]
        k3 = dist[0][4]
        
        
        u_correct = u * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6) + 2 * p1* u * v + p2 * (r ** 2 + 2 * u ** 2) 
        v_correct = v * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6) + p1 * (r ** 2 + 2 * v ** 2) + 2 * p2 * u * v  
        
              
        return [u_correct,v_correct]
    
    
        



        
    

