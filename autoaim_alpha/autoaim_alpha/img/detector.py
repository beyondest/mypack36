import cv2
import numpy as np
from typing import Union,Optional
from ..os_op.basic import *
from ..os_op.global_logger import *
from .const import *
from .tools import *
from ..os_op.decorator import *
from ..camera.mv_class import *
from ..utils_network.mymodel import *
from ..utils_network.actions import *
from .filter import *
from .depth_estimator import *
from ..utils_network.api_for_yolov5 import Yolov5_Post_Processor
########################################### Params ##############################################################

class Tradition_Params(Params):
    def __init__(self) -> None:
        super().__init__()
        
        self.gaussianblur_kernel_size = GAUSSIAN_BLUR_KERNAL_SIZE
        self.gaussianblur_x = GAUSSIAN_BLUR_X        
        self.close_kernel_size = CLOSE_KERNEL_SIZE
        self.red_armor_yuv_range = RED_ARMOR_YUV_RANGE
        self.blue_armor_yuv_range = BLUE_ARMOR_YUV_RANGE
        self.strech_max = STRECH_MAX
        self.red_armor_binary_roi_threshold = RED_ARMOR_BINARY_ROI_THRESHOLD
        self.blue_armor_binary_roi_threshold = BLUE_ARMOR_BINARY_ROI_THRESHOLD
        
class Net_Params(Params):
    
    def __init__(self) -> None:
        super().__init__()
        self.engine_type = 'ort'
        
        self.input_name = 'input'
        self.output_name = 'output'
        self.input_size =  NET_INPUT_SIZE
        self.input_dtype = NET_INPUT_DTYPE
        
        self.yolov5_input_name = 'images'
        self.yolov5_output_name = 'output0'
        self.yolov5_input_size = [640,640]
        self.yolov5_input_dtype = 'float32'
        
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.enemy_armor_index_list = [0,1]
        self.agnostic = False
        self.max_det = 20
        


############################################### Armor Detector #################################################################3    
        

class Armor_Detector:
    
    def __init__(self,
                 armor_color:str = 'red',
                 mode:str = 'Dbg',
                 tradition_config_folder :Union[str,None] = None,
                 net_config_folder :Union[str,None] = None,
                 save_roi_key:str = 'c',
                 depth_estimator_config_yaml:Union[str,None] = None,
                 if_yolov5:bool = True
                 ) -> None:
        
        CHECK_INPUT_VALID(armor_color,'red','blue')
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        self.mode = mode
        self.if_yolov5 = if_yolov5
        self.reset_result()
            
        self.net_detector = Net_Detector(
                                         mode=mode,
                                         net_config_folder=net_config_folder,
                                         if_yolov5=if_yolov5
                                         )
        self.depth_estimator = Depth_Estimator(depth_estimator_config_yaml,
                                            mode=mode
                                            )
        
        if self.if_yolov5:
            lr1.warn('Will apply yolov5 detect')
        else:
            lr1.warn('Will apply tradition detect')
        
            self.tradition_detector = Tradition_Detector(
                                                        armor_color,
                                                        mode,
                                                        tradition_config_folder_path=tradition_config_folder,
                                                        roi_single_shape=self.net_detector.params.input_size,
                                                        save_roi_key=save_roi_key
                                                        )  
        

        
    @timing(1)
    def get_result(self,img:np.ndarray)->Union[list,None]:
        """@timing(1)\n
        Get result of armor detection\n
        Input:
            img:
                if_yolov5:
                    img is BGR image with shape (640,640,3)
                else:
                    img is BGR image with shape > net_input_size
        Returns:
            list of dict or None
                big_rec: conts of big rec
                result: armor name
                probability: confidence of armor
                rvec:   rvec in camera frame, notice xyz is in rviz frame
                pos:    tvec in camera frame, notice xyz is in rviz frame
                
        """
        
        self.reset_result()
        
        if self.if_yolov5:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            self._net_part(img)
            lr1.warn('yolov5 need RGB, may be faster if you set camera output RGB')
            
        else:
            self._tradition_part(img)
            self._net_part()
            self._filter_part()                        
        
        self._depth_and_final_part()
        
        if len(self.final_result_list) > 0:
            if self.mode == 'Dbg':
                lr1.debug(f'Detector : Final result nums: {len(self.final_result_list)}')
            return self.final_result_list
        else:
            if self.mode == 'Dbg':
                lr1.debug('Detector : Final result nums: 0')
            return None

    
    def visualize(self,
                  img,
                  fps,
                  windows_name:Union[str,None] = 'detect_result',
                  fire_times:int = 0,
                  target_abs_pitch:float = 0.0,
                  target_abs_yaw:float = 0.0
                 )->None:
        
        """visualize the result of armor detection,
            if windows_name is None, return img drawn result on it but not show """
        add_text(img,'FPS',fps,color=(255,255,255),scale_size=0.8)  
            
        if self.final_result_list:
            lr1.debug(f'Final result nums: {len(self.final_result_list)}')
            
            for i in self.final_result_list:
                draw_big_rec_list([i['big_rec']],img,color=(0,255,0))
                add_text(   img,
                            f'pro:{i["probability"]:.2f}',
                            value=i['result'],
                            pos=i['big_rec'][0],
                            color=(0,0,255),
                            scale_size=0.7)
                add_text(   img,
                            f'tx:{i["pos"][0]:.4f}',
                            value=f'y:{i["pos"][1]:.4f} z:{i["pos"][2]:.4f}',
                            pos=(20,20),
                            color=(0,0,255),
                            scale_size=0.7)
                add_text(   img,
                            f'rx:{i["rvec"][0]:.4f}',
                            value=f'y:{i["rvec"][1]:.4f} z:{i["rvec"][2]:.4f}',
                            pos=(20,100),
                            color=(0,0,255),
                            scale_size=0.7)
                add_text(   img,
                            f'fire:{fire_times}',
                            value=f'tar_pit:{target_abs_pitch:.4f},tar_yaw:{target_abs_yaw:.4f}',
                            pos=(20,150),
                            color=(0,0,255),
                            scale_size=0.7)
                
        if windows_name is None:
            return img   
        
        cv2.imshow(windows_name,img)
        cv2.waitKey(1)
    
            
         
    def reset_result(self):
        self.roi_single_list = None
        self.big_rec_list = None
        self.probability_list = None
        self.result_list = None
        self.final_result_list = None
        
    
    def _tradition_part(self,img_bgr:np.ndarray):
        
        if img_bgr is None :
            lr1.warning("IMG : No img to apply tradition part")
            return None
      
        [self.roi_single_list,self.big_rec_list], tradition_time= self.tradition_detector.get_output(img_bgr)
        
        if self.mode == 'Dbg':
            lr1.debug(f'Tradition Time : {tradition_time:.6f}')
            
            if self.big_rec_list is not None:
                lr1.debug(f'Tradition Find Target Nums : {len(self.big_rec_list)}')
            else:
                lr1.debug(f"Tradition Find Nothing")
                
    def _net_part(self,img_rgb_640:Union[np.ndarray,None]=None):
        if self.if_yolov5: 
            
            (big_rec_list,self.probability_list,self.result_list), net_time =self.net_detector.get_output([img_rgb_640])
            self.big_rec_list = expand_rec_wid(big_rec_list,
                                               expand_rate=self.depth_estimator.pnp_params.expand_rate,
                                               img_size_yx=img_rgb_640.shape[:2])
            
        else:
            if self.roi_single_list is None:
                if self.mode == 'Dbg':
                    lr1.debug("IMG : No img to apply net part")
                return None
            
            else:
                
                tmp_list,net_time = self.net_detector.get_output(self.roi_single_list)
                if tmp_list is not None:  
                    self.probability_list,self.result_list = tmp_list
        
        if self.mode == 'Dbg':
            lr1.debug(f"Net Time : {net_time:.6f}")
            if self.probability_list is not None:
                lr1.debug(f'Net Find Target : {self.result_list}')
            else:
                lr1.debug('Net Find Nothing')    
        
            
    def _filter_part(self):
        """apply confidence filter and depth estimation to get final result
        """
        pro_list =[]
        big_rec_list = []
        result_list = []
        if self.probability_list is not None:
            for i in range(len(self.probability_list)):
                if self.probability_list[i] > self.net_detector.params.conf_thres:
                    pro_list.append(self.probability_list[i])
                    big_rec_list.append(self.big_rec_list[i])
                    result_list.append(self.result_list[i])
                    
        self.probability_list = pro_list
        self.big_rec_list = big_rec_list
        self.result_list = result_list
        
        if self.mode == 'Dbg':
            lr1.debug(f'Confience Filter Target Nums : {len(self.probability_list)}')

            
    def _depth_and_final_part(self):
        self.final_result_list = []
        for i,name in enumerate(self.result_list):
        
            obj_class = 'small' if name in self.depth_estimator.pnp_params.small_armor_name_list else 'big'
            depth_info = self.depth_estimator.get_result((self.big_rec_list[i],obj_class))
            if depth_info is None:
                lr1.warning(f"Depth Estimator Fail to get result, skip this target {self.big_rec_list[i]} {name}")
                continue
            
            each_result = {
                                    'result':self.result_list[i],
                                    'probability':self.probability_list[i],
                                    'big_rec':self.big_rec_list[i],
                                    'pos':depth_info[0],
                                    'rvec':depth_info[1]
                                    }   
            
            self.final_result_list.append(each_result)
        
        
    
            
############################################## tradition Detector#######################################################    
    

class Tradition_Detector:
    def __init__(self,
                 armor_color:str,
                 mode:str,
                 roi_single_shape:list,
                 tradition_config_folder_path:Union[str,None] = None,
                 save_roi_key:str = 'c'
                 ) -> None:
        
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        CHECK_INPUT_VALID(armor_color,'red','blue')
        
        
        self.mode = mode
        self.filter1 = Filter_of_lightbar(mode)
        self.filter2 = Filter_of_big_rec(mode)
        self.Tradition_Params = Tradition_Params()
        self.armor_color = armor_color
        self.roi_single_shape =roi_single_shape
        self.if_enable_save_roi = False
        self.if_enable_preprocess_config = False
                
        if tradition_config_folder_path is not None:
            self.load_params_from_folder(tradition_config_folder_path)
        
        
        if self.mode == 'Dbg':
            cv2.namedWindow('single',cv2.WINDOW_FREERATIO)
            cv2.namedWindow('roi_transform',cv2.WINDOW_FREERATIO)
            cv2.namedWindow('roi_binary',cv2.WINDOW_FREERATIO)
            self.roi_single = None
        self.save_roi_key = save_roi_key
        
    @timing(1)
    def get_output(self,img_bgr:np.ndarray)->Union[list,None]:
        """@timing

        Input:
            img_bgr ,shape is (512,640,3)
        Returns:
            [roi_binary_list,big_rec_list] (they are in same length)
        Notice:
            if Dbg,will draw on img_bgr
            
        """
        if img_bgr is None:
            lr1.warning("IMG : tradition detector get None img")
        
            return None
        if self.if_enable_preprocess_config:
            self._detect_trackbar_config()
        # No change to img_bgr
        img_single, preprocess_time1 = self._pre_process_bgr1(img_bgr)
        
        # Draw small cont to img_bgr
        big_rec_list,find_big_rec_time = self._find_big_rec(img_single,None)
        
        # No change to img_bgr
        roi_transform_list , pickup_roi_transform_time = self._pickup_roi_transform(big_rec_list,img_bgr)
        
        
        roi_binary_list, binary_roi_time = self._binary_roi_transform_list(roi_transform_list)
        
        #center_list = turn_big_rec_list_to_center_points_list(big_rec_list)
        
        if self.mode == 'Dbg':
            lr1.debug(f'pre_process1_time : {preprocess_time1:.4f}, find_big_rec_time : {find_big_rec_time:.4f}, pickup_roi_transfomr_time : {pickup_roi_transform_time:.4f}, binary_roi_time : {binary_roi_time:.4f}')
            cv2.imshow('single',img_single)
            
            if big_rec_list is not None and len(big_rec_list)>0:
                combined_roi_transform = roi_transform_list[0]
                combined_roi_binary = roi_binary_list[0]
                for i in range(1,len(roi_transform_list)):
                    combined_roi_transform = np.r_[combined_roi_transform,roi_transform_list[i]]
                    combined_roi_binary = np.r_[combined_roi_binary,roi_binary_list[i]]
                    
                cv2.imshow(f'roi_transform',combined_roi_transform)
                cv2.imshow(f'roi_binary',combined_roi_binary) 
                self.roi_single = roi_binary_list[0]
        
        if self.if_enable_save_roi:
            
            if roi_binary_list:
                self.save_img_count +=1
                if self.save_img_count % self.save_interval == 0:
                    for i in range(len(roi_binary_list)):
                        
                        t = time.time()
                        save_path_bin = os.path.join(self.save_folder,'bin',f'{t}.jpg')
                        save_path_transform = os.path.join(self.save_folder,'transform',f'{t}.jpg')
                        
                        
                        cv2.imwrite(save_path_bin,roi_binary_list[i])
                        trans = cv2.cvtColor(roi_transform_list[i],cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(save_path_transform,trans)
                    
                    
            
        return [roi_binary_list,big_rec_list]
    
    def save_params_to_folder(self,tradition_config_folder_path:str = './tmp_tradition_config')->None:
        
        filter1_yaml_path = os.path.join(tradition_config_folder_path,self.armor_color,'filter1_params.yaml')
        filter2_yaml_path = os.path.join(tradition_config_folder_path,self.armor_color,'filter2_params.yaml')
        preprocess_yaml_path = os.path.join(tradition_config_folder_path,self.armor_color,'preprocess_params.yaml')
        
        if not os.path.exists(tradition_config_folder_path):
            os.makedirs(tradition_config_folder_path)
        if not os.path.exists(os.path.join(tradition_config_folder_path,self.armor_color)):
            os.makedirs(os.path.join(tradition_config_folder_path,self.armor_color))
        
        self.filter1.filter_params.save_params_to_yaml(filter1_yaml_path)
        self.filter2.filter_params.save_params_to_yaml(filter2_yaml_path)
        self.Tradition_Params.save_params_to_yaml(preprocess_yaml_path)
        
        lr1.info(f'IMG : Save tradition params success : {tradition_config_folder_path}')
        
    
    def load_params_from_folder(self,tradition_confit_folder_path:str):
        
        CHECK_INPUT_VALID(os.path.exists(tradition_confit_folder_path),True)
        CHECK_INPUT_VALID(os.listdir(tradition_confit_folder_path),['red','blue'],['red'],['blue'])
        
        root_path = os.path.join(tradition_confit_folder_path,self.armor_color)
        preprocess_path = os.path.join(root_path,'preprocess_params.yaml')
        filter1_path = os.path.join(root_path,'filter1_params.yaml')
        filter2_path = os.path.join(root_path,'filter2_params.yaml')
        
        
        self.Tradition_Params.load_params_from_yaml(preprocess_path)
        lr1.info(f'IMG : Load preprocess params success : {preprocess_path}')
        
        self.filter1.filter_params.load_params_from_yaml(filter1_path)
        lr1.info(f'IMG : Load filter1 params success : {filter1_path}')
        
        self.filter2.filter_params.load_params_from_yaml(filter2_path)
        lr1.info(f'IMG : Load filter2 params success : {filter2_path}')
        
                
    def enable_preprocess_config(self,window_name:str = 'preprocess_config',press_key_to_save:str = 's'):
        self.config_window_name =window_name
        self.press_key_to_save = press_key_to_save
        self.if_enable_preprocess_config = True
        def for_trackbar(x):
            pass
        cv2.namedWindow(window_name,cv2.WINDOW_FREERATIO)
        cv2.createTrackbar('yuv_range_min',window_name,0,255,for_trackbar)
        cv2.createTrackbar('yuv_range_max',window_name,0,255,for_trackbar)
        cv2.createTrackbar('threshold',window_name,0,255,for_trackbar)
        cv2.createTrackbar('kernel_wid',window_name,0,255,for_trackbar)
        cv2.createTrackbar('kernel_hei',window_name,0,255,for_trackbar)
        

        cv2.setTrackbarPos('kernel_wid',window_name,self.Tradition_Params.close_kernel_size[1])
        cv2.setTrackbarPos('kernel_hei',window_name,self.Tradition_Params.close_kernel_size[0])
        
        
        if self.armor_color == 'red':
            cv2.setTrackbarPos('yuv_range_min',window_name,self.Tradition_Params.red_armor_yuv_range[0])
            cv2.setTrackbarPos('yuv_range_max',window_name,self.Tradition_Params.red_armor_yuv_range[1])
            cv2.setTrackbarPos('threshold',window_name,self.Tradition_Params.red_armor_binary_roi_threshold)
            
        if self.armor_color == 'blue':
            cv2.setTrackbarPos('yuv_range_min',window_name,self.Tradition_Params.blue_armor_yuv_range[0])
            cv2.setTrackbarPos('yuv_range_max',window_name,self.Tradition_Params.blue_armor_yuv_range[1])
            cv2.setTrackbarPos('threshold',window_name,self.Tradition_Params.blue_armor_binary_roi_threshold)
            
    
    def _detect_trackbar_config(self):
        
        self.Tradition_Params.close_kernel_size[0] = cv2.getTrackbarPos('kernel_wid',self.config_window_name)
        self.Tradition_Params.close_kernel_size[1] = cv2.getTrackbarPos('kernel_hei',self.config_window_name)
        
        
        if self.armor_color ==  'red':
            
            self.Tradition_Params.red_armor_yuv_range[0] = cv2.getTrackbarPos('yuv_range_min',self.config_window_name)
            self.Tradition_Params.red_armor_yuv_range[1] = cv2.getTrackbarPos('yuv_range_max',self.config_window_name)
            self.Tradition_Params.red_armor_binary_roi_threshold = cv2.getTrackbarPos('threshold',self.config_window_name)
        else:
            
            self.Tradition_Params.blue_armor_yuv_range[0] = cv2.getTrackbarPos('yuv_range_min',self.config_window_name)
            self.Tradition_Params.blue_armor_yuv_range[1] = cv2.getTrackbarPos('yuv_range_max',self.config_window_name)
            self.Tradition_Params.blue_armor_binary_roi_threshold = cv2.getTrackbarPos('threshold',self.config_window_name)
            
            
        if cv2.waitKey(1) == ord(self.press_key_to_save):
            
            self.Tradition_Params.save_params_to_yaml('preprocess_params.yaml')
        
        if cv2.waitKey(1) == ord(self.save_roi_key):

            if self.roi_single is not None:

                cv2.imwrite('roi_tmp.png',self.roi_single)
        
            
    def enable_save_roi(self,save_folder:str = 'roi_binary',img_suffix:str = 'jpg',save_interval:int = 10):
       
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(os.path.join(save_folder,'bin')):
            os.makedirs(os.path.join(save_folder,'bin'))
        if not os.path.exists(os.path.join(save_folder,'transform')):
            os.makedirs(os.path.join(save_folder,'transform'))
        
        self.save_folder = save_folder
        self.if_enable_save_roi = True
        self.img_suffix = img_suffix
        self.save_interval = save_interval
        self.save_img_count = 0
    
    @timing(1)
    def _find_big_rec(self,img_single:np.ndarray,img_bgr:Union[np.ndarray,None] = None)->Union[list,None]:
        """@timing

        Args:
            img_single (np.ndarray): _description_

        Returns:
            Union[list,None]: _description_
        """
        
        if img_single is None:
            return None
        
        conts,arrs = cv2.findContours(img_single,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        
        
        small_rec_pairs_list = self.filter1.get_output(conts,img_bgr=img_bgr)
        
        
        #big_rec_list = [make_big_rec(rec_pair[0],rec_pair[1]) for rec_pair in small_rec_pairs_list] if small_rec_pairs_list is not None else None 
        #big_rec_list = expand_rec_wid(big_rec_list,EXPAND_RATE,img_size_yx=img_single.shape)
        big_rec_list = [get_trapezoid_corners(rec_pair[0], rec_pair[1]) for rec_pair in small_rec_pairs_list] if small_rec_pairs_list is not None else None 
        big_rec_list = expand_trapezoid_wid(big_rec_list,EXPAND_RATE,img_size_yx=img_single.shape)
        
        big_rec_list = self.filter2.get_output(big_rec_list)
            
        return big_rec_list
    

    @timing(1)
    def _pre_process_bgr1(self,img_bgr:np.ndarray):
        """@timing

        Args:
            img_bgr (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        
        if img_bgr is None:
            return None
        
        out = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2YUV)
        out = cv2.GaussianBlur(out,
                               self.Tradition_Params.gaussianblur_kernel_size,
                               self.Tradition_Params.gaussianblur_x)
        y,u,v = cv2.split(out)
        
        if self.armor_color == 'red':
            out = cv2.inRange(v.reshape(img_bgr.shape[0],img_bgr.shape[1],1),
                              self.Tradition_Params.red_armor_yuv_range[0],
                              self.Tradition_Params.red_armor_yuv_range[1]
                              )
        
        else:
            out = cv2.inRange(u.reshape(img_bgr.shape[0],img_bgr.shape[1],1),
                              self.Tradition_Params.blue_armor_yuv_range[0],
                              self.Tradition_Params.blue_armor_yuv_range[1]
                              )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           self.Tradition_Params.close_kernel_size)
        out = cv2.morphologyEx(out,cv2.MORPH_CLOSE,kernel)
        
        return out
        
        
    @timing(1)
    def _binary_roi_transform_list(self,roi_transform_list:list)->Union[np.ndarray,None]:
        """@timing

        Args:
            roi_transform_list (list): _description_

        Returns:
            Union[np.ndarray,None]: _description_
        """
        if roi_transform_list is None:
            return None

        roi_single_list=[]
        for i in roi_transform_list:
            
            
            dst=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
            dst=gray_stretch(dst,255)
            
            if self.roi_single_shape ==[32,32]:
                exclude_light_bar_region = dst[:,5:27]
            elif self.roi_single_shape ==[64,64]:
                exclude_light_bar_region = dst[:,10:54]
            else:
                raise ValueError(f'roi_single_shape Error {self.roi_single_shape}, only support (32,32) and (64,64)')

            thresh = get_threshold(exclude_light_bar_region,255,mode=self.mode)
            if self.if_enable_preprocess_config:
                thresh = self.Tradition_Params.blue_armor_binary_roi_threshold if self.armor_color == 'blue' else self.Tradition_Params.red_armor_binary_roi_threshold
            
            ret,dst =cv2.threshold(dst,thresh,255,cv2.THRESH_BINARY) 
            roi_single_list.append(dst)
        
        return roi_single_list
    
    
    @timing(1)
    def _pickup_roi_transform(self,big_rec_list:list,img_bgr:np.ndarray):
        """@timing

        Args:
            big_rec_list (list): _description_
            img_bgr (np.ndarray): _description_

        Returns:
            _type_: _description_
        """


        if big_rec_list is None:
            return None
        
        roi_transform_list=[]
        
        for i in big_rec_list:
            
            _,_,wid,hei,_,_=getrec_info(i)
            
            i=order_rec_points(i)
            dst_points=np.array([[0,0],[wid-1,0],[wid-1,hei-1],[0,hei-1]],dtype=np.float32)
            M=cv2.getPerspectiveTransform(i.astype(np.float32),dst_points)
            dst=cv2.warpPerspective(img_bgr,M,(int(wid),int(hei)),flags=cv2.INTER_LINEAR)
            
            
            dst = cv2.resize(dst,self.roi_single_shape)
            
            roi_transform_list.append(dst)
        
        return roi_transform_list
        
    
        
        
        
  

############################################# Net Detector########################################################3


class Net_Detector:
    def __init__(self,
                 net_config_folder :str,
                 mode:str = 'Dbg',
                 if_yolov5:bool = True
                 ) -> None:
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        
        self.mode = mode

        self.if_yolov5 = if_yolov5
        self.params = Net_Params()
        
        self.load_params_from_folder(net_config_folder)
        if self.if_yolov5:
            self.yolov5_post_processor = Yolov5_Post_Processor(self.class_info,
                                                               self.params.conf_thres,
                                                               self.params.iou_thres,
                                                               self.params.enemy_armor_index_list,
                                                               self.params.agnostic,
                                                               multi_label=False,
                                                               labels=(),
                                                               max_det=self.params.max_det,
                                                               mode=self.mode
                                                               )
            
        
        
        if self.params.engine_type == 'ort':
            if self.if_yolov5:
                self.input_dtype = np.float32 if self.params.yolov5_input_dtype == 'float32' else np.float16
            else:
                self.input_dtype = np.float32 if self.params.input_dtype == 'float32' else np.float16
                
                
            self.engine = Onnx_Engine(self.model_path,if_offline=True)
            self.onnx_inputname = self.params.yolov5_input_name if self.if_yolov5 else self.params.input_name
            self.onnx_outputname = self.params.yolov5_output_name if self.if_yolov5 else self.params.output_name

        elif self.params.engine_type == 'trt':
            if self.if_yolov5:
                self.input_dtype = np.float32 if self.params.yolov5_input_dtype == 'float32' else np.float16
            else:
                self.input_dtype = np.float32 if self.params.input_dtype == 'float32' else np.float16
                
            self.engine = TRT_Engine_2(self.model_path,
                                       max_batchsize=MAX_INPUT_BATCHSIZE)
        
            

    def save_params_to_yaml(self,yaml_path:str = './tmp_net_params.yaml'):
        
        self.params.save_params_to_yaml(yaml_path)
        
    
    def load_params_from_folder(self,folder_path:str):
        
        net_config_path = os.path.join(folder_path,'net_params.yaml')
        
        self.params.load_params_from_yaml(net_config_path)
        if self.if_yolov5:
            class_path = os.path.join(folder_path,'yolov5_class.yaml')
        else:
            class_path = os.path.join(folder_path,'classifier_class.yaml')
        
        self.class_info = Data.get_file_info_from_yaml(class_path)
        
        if self.params.engine_type == 'ort':
            if self.if_yolov5:
                self.model_path = os.path.join(folder_path,'yolov5.onnx')
            else:
                self.model_path = os.path.join(folder_path,'classifier.onnx')
        
        elif self.params.engine_type == 'trt':
            if self.if_yolov5:
                self.model_path = os.path.join(folder_path,'yolov5.trt')
            else:
                self.model_path = os.path.join(folder_path,'classifier.trt')
        
        else: 
            raise ValueError(f'Engine Type Error {self.params.engine_type}, only support ort and trt')
        
    @timing(1)
    def get_output(self,
                   input_list:Union[list,None]
                   )->Union[list,None]:
        """@timing

        Input:
            yolov5:
                input_list: [img_rgb1,img_rgb2,...]
            classifier:
                input_list: [roi_single1,roi_single2,...]

        Returns:
            yolov5:
                [conts_list, pro_list, cls_name_list]
            classifier:
                [[probability1,probability2,...],[armor_type1,armortype2,...]]
            None:
                If input_list is None or len(input_list) == 0
        """
        if input_list is None or len(input_list) == 0:
            return None
            
        
        inp = normalize_to_nparray(input_list,dtype=self.input_dtype)
        if self.if_yolov5:
            inp = np.transpose(inp,(0,3,1,2))
        
        if self.params.engine_type == 'ort':
            
            model_output,ref_time =self.engine.run(output_nodes_name_list=None,
                            input_nodes_name_to_npvalue={self.onnx_inputname:inp})
            
            if self.if_yolov5:
                
                out, post_pro_t = self.yolov5_post_processor.get_output(model_output[0])
                
            else:
                out,post_pro_t =  self._classifier_post_process(model_output[0]) 
            
            if self.mode == 'Dbg':
                lr1.debug(f'Refence Time: {ref_time:.5f}, Post Process Time: {post_pro_t:.5f}')
            
            return out
        
        elif self.params.engine_type == 'trt':
            
            model_output,ref_time = self.engine.run({0:inp})
            
            if self.if_yolov5:
                out, post_pro_t = self.yolov5_post_processor.get_output(model_output[0])
                
            else:
                out,post_pro_t =  self._classifier_post_process(model_output[0]) 
            
            if self.mode == 'Dbg':
                lr1.debug(f'Refence Time: {ref_time:.5f}, Post Process Time: {post_pro_t:.5f}')
                
            return out 
     
    @timing(1)   
    def _classifier_post_process(self,
                                 model_model_output0):
        """@timing(1)

        Args:
            model_model_output (_type_): _description_

        Returns:
            [probabilities_list,result_list]
        """
        
        probabilities_list,index_list = trans_logits_in_batch_to_result(model_model_output0)

        result_list = [self.class_info[i] for i in index_list]
                
        return [probabilities_list,result_list]
            
            
            
            
