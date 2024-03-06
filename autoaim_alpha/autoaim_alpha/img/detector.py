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
        self.input_name = 'inputs'
        self.output_name = 'outputs'
        self.input_size =  NET_INPUT_SIZE
        self.input_dtype = NET_INPUT_DTYPE
        self.confidence = NET_CONFIDENCE
        
        


############################################### Armor Detector #################################################################3    
        

class Armor_Detector:
    
    def __init__(self,
                 armor_color:str = 'red',
                 mode:str = 'Dbg',
                 tradition_config_folder :Union[str,None] = None,
                 net_config_folder :Union[str,None] = None,
                 save_roi_key:str = 'c',
                 depth_estimator_config_yaml:Union[str,None] = None
                 ) -> None:
        
        CHECK_INPUT_VALID(armor_color,'red','blue')
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        self.mode = mode
        
        self.reset_result()
        
        
        self.net_detector = Net_Detector(
                                         mode=mode,
                                         net_config_folder=net_config_folder
                                         )
    
        
        self.tradition_detector = Tradition_Detector(
                                                       armor_color,
                                                       mode,
                                                       tradition_config_folder_path=tradition_config_folder,
                                                       roi_single_shape=self.net_detector.params.input_size,
                                                       save_roi_key=save_roi_key
                                                       )  
          
        self.depth_estimator = Depth_Estimator(depth_estimator_config_yaml,
                                                mode=mode
                                                )
        
    @timing(1)
    def get_result(self,img_bgr:np.ndarray,img_bgr_exposure2:np.ndarray)->Union[list,None]:
        """@timing(1)\n
        Get result of armor detection\n
        Returns:
            Union[list,None]:
            
                if success,return a list of dict,each dict contains\n
                'big_rec'
                'center'
                'result'
                'probability'
                'rvec'
                'pos'\n
                
                if fail,return None
                
        """
        self.reset_result()
        self._tradition_part(img_bgr,img_bgr_exposure2)
        
        self._net_part()
        self._filter_part()
        
        if self.success_flag == True:
            if self.mode == 'Dbg':
                
                pass
            else:
                pass
            
            return self.final_result_list
        else:
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
                            pos=(round(i['center'][0]+20),round(i['center'][1])+20),
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
        self.center_list = None
        self.roi_single_list = None
        self.big_rec_list = None
        self.probability_list = None
        self.result_list = None
        self.final_result_list = None
        self.success_flag = False
        
        
    
    
    def _tradition_part(self,img_bgr:np.ndarray,img_bgr_exposure2:np.ndarray):
        
        if img_bgr is None or img_bgr_exposure2 is None:
            lr1.warning("IMG : No img to apply tradition part")
            return None
      
        tmp_list, tradition_time= self.tradition_detector.get_output(img_bgr,img_bgr_exposure2)
        self.center_list,self.roi_single_list,self.big_rec_list  = tmp_list
        
        if self.mode == 'Dbg':
            lr1.debug(f'Tradition Time : {tradition_time:.6f}')
            
            if self.center_list is not None:
                lr1.debug(f'Tradition Find Target : {len(self.center_list)}')
            else:
                lr1.debug(f"Tradition Find Nothing")
                
    def _net_part(self):
        if self.roi_single_list is None:
            if self.mode == 'Dbg':
                lr1.debug("IMG : No img to apply net part")
            return None
        
        
        
        tmp_list,net_time = self.net_detector.get_output(self.roi_single_list)  
        self.probability_list,self.result_list = tmp_list   
        
        if self.mode == 'Dbg':
            lr1.debug(f"Net Time : {net_time:.6f}")
            if self.probability_list is not None:
                lr1.debug(f'Net Find Target : {len(self.probability_list)}')
            else:
                lr1.debug('Net Find Nothing')    
        
            
    def _filter_part(self):
        """apply confidence filter and depth estimation to get final result
        """
        
        self.final_result_list = []
        
        if self.probability_list is not None:
            
            for i in range(len(self.probability_list)):
                if self.probability_list[i] > self.net_detector.params.confidence:
                    
                    obj_class = 'small' if self.result_list[i] in ['2x','3x','4x','5x','basex','sentry'] else 'big'
                    
                    output = self.depth_estimator.get_result((self.big_rec_list[i],obj_class))
                    
                    if output is None:
                        lr1.warning(f"Depth Estimator Fail to get result, skip this target")
                        continue
                    else:
                        pos,rvec = output
                        
                    each_result = {'pos':pos,
                                   'center':self.center_list[i],
                                   'result':self.result_list[i],
                                   'probability':self.probability_list[i],
                                   'big_rec':self.big_rec_list[i],
                                   'rvec':rvec}
                    
                    self.final_result_list.append(each_result)
            
            if len(self.final_result_list)>0:
                self.success_flag = True
                
                #sorted(self.final_result_list,key=lambda x:x['probability'],reverse=True)
                
            else:
                self.success_flag = False
        else:
            self.success_flag = False
            
   
       
            
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
    def get_output(self,img_bgr:np.ndarray,img_bgr_exposure2:np.ndarray)->Union[list,None]:
        """@timing

        Input:
            img_bgr,img_bgr_in_exposure2 ,shape is (512,640,3)
        Returns:
            [center_list,roi_binary_list,big_rec_list] (they are in same length)
        Notice:
            if Dbg,will draw on img_bgr
            
        """
        if img_bgr is None or img_bgr_exposure2 is None:
            lr1.warning("IMG : tradition detector get None img")
        
            return None
        if self.if_enable_preprocess_config:
            self._detect_trackbar_config()
        # No change to img_bgr
        img_single, preprocess_time1 = self._pre_process_bgr1(img_bgr)
        
        # Draw small cont to img_bgr
        big_rec_list,find_big_rec_time = self._find_big_rec(img_single,None)
        
        # No change to img_bgr
        roi_transform_list , pickup_roi_transform_time = self._pickup_roi_transform(big_rec_list,img_bgr_exposure2)
        
        
        roi_binary_list, binary_roi_time = self._binary_roi_transform_list(roi_transform_list)
        
        center_list = turn_big_rec_list_to_center_points_list(big_rec_list)
        
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
                    
                    
            
        return [center_list,roi_binary_list,big_rec_list]
    
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
    def _pickup_roi_transform(self,big_rec_list:list,img_bgr_exposure2:np.ndarray):
        """@timing

        Args:
            big_rec_list (list): _description_
            img_bgr_exposure2 (np.ndarray): _description_

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
            dst=cv2.warpPerspective(img_bgr_exposure2,M,(int(wid),int(hei)),flags=cv2.INTER_LINEAR)
            
            
            dst = cv2.resize(dst,self.roi_single_shape)
            
            roi_transform_list.append(dst)
        
        return roi_transform_list
        
    
        
        
        
  

############################################# Net Detector########################################################3


class Net_Detector:
    def __init__(self,
                 net_config_folder :str,
                 mode:str = 'Dbg'
                 ) -> None:
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        
        self.mode = mode
        self.params = Net_Params()
        
        self.load_params_from_folder(net_config_folder)
        
        if self.params.engine_type == 'ort':
            self.input_dtype = np.float32 if self.params.input_dtype == 'float32' else np.float16
            
            self.engine = Onnx_Engine(self.model_path,if_offline=True)
            self.onnx_inputname = self.params.input_name
            self.onnx_outputname = self.params.output_name

        elif self.params.engine_type == 'trt':
            
            self.input_dtype = np.float32 if self.params.input_dtype == 'float32' else np.float16
            self.engine = TRT_Engine_2(self.model_path,
                                       max_batchsize=MAX_INPUT_BATCHSIZE)
        
            

    def save_params_to_yaml(self,yaml_path:str = './tmp_net_params.yaml'):
        
        self.params.save_params_to_yaml(yaml_path)
        
    
    def load_params_from_folder(self,folder_path:str):
        
        class_path = os.path.join(folder_path,'class.yaml')
        net_config_path = os.path.join(folder_path,'net_params.yaml')
        
        self.params.load_params_from_yaml(net_config_path)
        self.class_info = Data.get_file_info_from_yaml(class_path)
        
        if self.params.engine_type == 'ort':
            self.model_path = os.path.join(folder_path,'model.onnx')
        
        elif self.params.engine_type == 'trt':
            self.model_path = os.path.join(folder_path,'model.trt')
        
        else: 
            raise ValueError(f'Engine Type Error {self.params.engine_type}, only support ort and trt')
        
    @timing(1)
    def get_output(self,
                   input_list:Union[list,None]
                   )->Union[list,None]:
        """@timing

        Input:
            [roi_single1,roi_single2,...]

        Returns:
            [[probability1,probability2,...],[armor_type1,armortype2,...]]
        """
        if input_list is None:
            return [None,None]
        
        if len(input_list) == 0 :
            return [None,None]
        
        
        
        inp = normalize_to_nparray(input_list,dtype=self.input_dtype)
        
        if self.params.engine_type == 'ort':
            
           
            output,ref_time =self.engine.run(output_nodes_name_list=None,
                            input_nodes_name_to_npvalue={self.onnx_inputname:inp})
            probabilities_list,index_list = trans_logits_in_batch_to_result(output[0])
            result_list = [self.class_info[index] for index in index_list]
            
            if self.mode == 'Dbg':
                lr1.debug(f'Refence Time: {ref_time:.5f}')
                
            return [probabilities_list,result_list]
        
        
        elif self.params.engine_type == 'trt':
            
            output,ref_time = self.engine.run({0:inp})
            probabilities_list,index_list = trans_logits_in_batch_to_result(output[0])
            result_list = [self.class_info[index] for index in index_list]
            
            if self.mode == 'Dbg':
                lr1.debug(f'Refence Time: {ref_time:.5f}')
                
            return [probabilities_list,result_list]
            
            
            
