
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
 
  









################################################Depth Estimator################################################3

class Depth_Estimator:
    def __init__(self,
                 depth_config_folder :Union[str,None],
                 mode:str = 'Dbg',
                 method_name:str = 'pnp'):
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        CHECK_INPUT_VALID(method_name,'pnp','opticalflow')
        
        self.mode = mode
        self.method = method_name
        
        
        if self.method == 'pnp':
            self.pnp_params = PNP_Params()
            if depth_config_folder is not None:
                if not os.path.exists(os.path.join(depth_config_folder,'pnp_params.yaml')):
                    lr1.critical(f'PnP params file not found in {depth_config_folder}')
                    raise ValueError(f'PnP params file not found in {depth_config_folder}')
                self.pnp_params.load_params_from_yaml(os.path.join(depth_config_folder,'pnp_params.yaml'))
            
            
            
    def get_result(self,input):
        """

        Returns:
            x,y,z: the position of the object in the camera coordinate system.
            
        Method:
            pnp: Perspective-n-Point algorithm.
                input: (big_rec, obj_wid_in_world,obj_hei_in_world): the coordinates of the object in the image.
                output: [x,y,z],rvecs: the position of the object in the camera coordinate system and the rotation matrix.
                
            pnp_fix: PnP algorithm with fixed parameters.
                input: (big_rec, obj_class = 'big' or'small'): the coordinates of the object in the image and the class of the object.
                output: [x,y,z],rvecs: the position of the object in the camera coordinate system and the rotation matrix.
                
        """
        
        
        if self.method == "pnp":
            big_rec = input[0]
            obj_wid_in_world = input[1]
            obj_hei_in_world = input[2]
            obj_pts = np.array([[-obj_wid_in_world/2, -obj_hei_in_world/2, 0],
                                [obj_wid_in_world/2, -obj_hei_in_world/2, 0],
                                [-obj_wid_in_world/2, obj_hei_in_world/2, 0],
                                [obj_wid_in_world/2, obj_hei_in_world/2, 0]], dtype=np.double)
            
            return self._PNP(big_rec,obj_pts)
        
        elif self.method == 'pnp_fix':
            big_rec = input[0]
            obj_class = input[1]
            if obj_class == 'big':
                obj_pts = self.pnp_params.big_obj_points
            elif obj_class =='small':
                obj_pts = self.pnp_params.small_obj_points
            else:
                lr1.critical(f'Invalid obj_class: {obj_class}')
                raise ValueError(f'Invalid obj_class: {obj_class}')
            
            return self._PNP(big_rec,obj_pts)
        
    def save_params_to_folder(self,folder_path:str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.method == 'pnp':
            self.pnp_params.save_params_to_yaml(os.path.join(folder_path,'pnp_params.yaml'))
    
    
    
    def _PNP(self,big_rec,obj_pts:np.ndarray):
        if big_rec is None:
            lr1.warning(f'big_rec is None')
            return None
        
        img_points = np.array(big_rec, dtype=np.double)
        img_points = img_points / self.pnp_params.img_shrink_scale
        
        
        success, rvecs, tvecs = cv2.solvePnP(obj_pts, img_points,
                                            np.array(self.pnp_params.mtx, dtype=np.double),
                                            np.array(self.pnp_params.dist, dtype=np.double))
        if not success:
            lr1.warning(f'PnP Failed, big_rec: {big_rec}, obj_pts: {obj_pts}')
            return None
        
        tvecs = np.array(tvecs)
        positions = [(tvecs[0][0], tvecs[1][0], tvecs[2][0])]
        x,y,z = positions[0]
        
        return [x,y,z],rvecs
        

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
        
        



        
    

