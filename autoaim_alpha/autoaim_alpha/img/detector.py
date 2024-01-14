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



########################################### Params ##############################################################


class Filter_Lightbar_Params(Params):
    def __init__(self) -> None:
        super().__init__()
        self.accept_area_range = SMALL_LIGHTBAR_SINGLE_AREA_RANGE
        self.accept_aspect_ratio_range = SMALL_LIGHTBAR_SINGLE_ASPECT_RATIO_RANGE
        self.accept_two_area_ratio_range = SMALL_LIGHTBAR_TWO_AREA_RATIO_RANGE
        self.accept_two_aspect_ratio_range = SMALL_LIGHTBAR_TWO_ASPECT_RATIO_RANGE
        self.accept_center_distance_range = SMALL_LIGHTBAR_CENTER_DISTANCE_RANGE

class Filter_Big_Rec_Params(Params):
    def __init__(self) -> None:
        super().__init__()
        self.accept_area_range =  BIG_REC_AREA_RANGE
        self.accept_aspect_ratio_range = BIG_REC_ASPECT_RATIO_RANGE

class PreProcess_Bgr_Params(Params):
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
        
    
    
############################################### Armor Detector #################################################################3    
        

class Armor_Detector:
    
    def __init__(self,
                 model_path:str,
                 engine_type:str,
                 class_yaml_path:str,
                 armor_color:str = 'red',
                 mode:str = 'Dbg',
                 confidence:float = 0.5
                 ) -> None:
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        self.mode = mode
        self.confidence = confidence
        self.center_list = []
        self.roi_single_list = []
        self.big_rec_list = []
        self.pro_list = []
        self.result_list = []
        
        # [[bigrec,center,result,pro],...]
        self.final_result_list = [] 
        # 
        self.success_flag = False
        
        self.tradition_detector = Traditional_Detector(
                                                       armor_color,
                                                       mode,
                                                       if_enable_filter2=True
                                                       )    
        
        self.net_detector = Net_Detector(model_path=model_path,
                                         class_yaml_path=class_yaml_path,
                                         mode=mode,
                                         engine_type=engine_type
                                         )
        
    def get_result(self,img_bgr:np.ndarray,img_bgr_exposure2:np.ndarray):
        self._tradition_part(img_bgr,img_bgr_exposure2)
        self._net_part()
        self._filter_part()
        if self.success_flag == True:
            if self.mode == 'Dbg':
                for i in self.final_result_list:
                    add_text(img_bgr,f'{i[2]}:',i[3],np.round(i[1
    
    def _tradition_part(self,img_bgr:np.ndarray,img_bgr_exposure2:np.ndarray):
        
        if img_bgr is None or img_bgr_exposure2 is None:
            lr1.warning("IMG : No img to apply tradition part")
            return None
      
        self.center_list,self.roi_single_list,self.big_rec_list , tradition_time= self.tradition_detector.get_output(img_bgr,img_bgr_exposure2)
        if self.mode == 'Dbg':
            print(f'Tradition Time : {tradition_time:.6f}')
            
            if self.center_list is not None:
                print(f'Tradition Find Target : {len(self.center_list)}')
            else:
                print(f"Tradition Find Nothing")
                
    def _net_part(self):
        if self.roi_single_list is None:
            lr1.warning("IMG : No img to apply net part")
            return None
        
        
        self.pro_list,self.result_list,net_time = self.net_detector.get_output(self.roi_single_list)     
           
        if self.mode == 'Dbg':
            print(f"Net Time : {net_time:.6f}")
            if self.pro_list is not None:
                print(f'Net Find Target : {len(self.pro_list)}')
            else:
                print('Net Find Nothing')    
            
    def _filter_part(self):
        self.final_result_list = []
        
        if self.pro_list is not None and len(self.pro_list) > 0:
            for i in range(len(self.pro_list)):
                if self.pro_list[i] > self.confidence:
                    
                    self.final_result_list.append(self.big_rec_list[i],self.center_list[i],self.result_list[i],self.pro_list[i])
            if len(self.final_result_list)>0:
                self.success_flag = True
            else:
                self.success_flag = False
        else:
            self.success_flag = False
            
            
            
############################################## Traditional Detector#######################################################    
    

class Traditional_Detector:
    def __init__(self,
                 armor_color:str,
                 mode:str,
                 if_enable_filter2:bool = True,
                 tradition_config_folder_path:Union[str,None] = None,
                 roi_single_shape:tuple = ROI_SINGLE_SHAPE,
                 save_roi_key:str = 'c'
                 ) -> None:
        
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        CHECK_INPUT_VALID(armor_color,'red','blue')
        
        
        self.mode = mode
        self.filter1 = Filter_of_lightbar(mode)
        self.filter2 = Filter_of_big_rec(mode)
        self.if_enable_filter2 = if_enable_filter2
        self.preprocess_bgr_params = PreProcess_Bgr_Params()
        self.armor_color = armor_color
        self.roi_single_shape =roi_single_shape
        if tradition_config_folder_path is not None:
            self.load_params_from_folder(tradition_config_folder_path)
        
        
        if self.mode == 'Dbg':
            cv2.namedWindow('single',cv2.WINDOW_FREERATIO)
            cv2.namedWindow('roi_transform',cv2.WINDOW_FREERATIO)
            cv2.namedWindow('roi_binary',cv2.WINDOW_FREERATIO)
            self.roi_single = None
            self.save_roi_key = save_roi_key
        
    @timing(1)
    def get_output(self,img_bgr:np.ndarray,img_bgr_exposure2:np.ndarray):
        """@timing

        Input:
            img_bgr,img_bgr_in_exposure2 ,shape is (512,640,3)
        Returns:
            center_list,roi_binary_list,big_rec_list (they are in same length)
        Notice:
            if Dbg,will draw on img_bgr
            
        """
        if img_bgr is None or img_bgr_exposure2 is None:
            lr1.warning("IMG : traditional detector get None img")
        
            return None
        
        img_single, preprocess_time1 = self._pre_process_bgr1(img_bgr)
        big_rec_list,find_big_rec_time = self._find_big_rec(img_single,img_bgr)
        roi_transform_list , pickup_roi_transform_time = self._pickup_roi_transform(big_rec_list,img_bgr_exposure2)
        roi_binary_list, binary_roi_time = self._binary_roi_transform_list(roi_transform_list)
        center_list = turn_big_rec_list_to_center_points_list(big_rec_list)
        
        if self.mode == 'Dbg':
            print('pre_process1_time',preprocess_time1)
            print('find_big_rec_time',find_big_rec_time)
            print('pickup_roi_transfomr_time',pickup_roi_transform_time)
            print('binary_time',binary_roi_time)
            cv2.imshow('single',img_single)
            draw_big_rec_list(big_rec_list,img_bgr)    
            draw_center_list(center_list,img_bgr)
            if big_rec_list is not None and len(big_rec_list) == 1:
                cv2.imshow('roi_transform',roi_transform_list[0])
                cv2.imshow('roi_binary',roi_binary_list[0])
                self.roi_single = roi_binary_list[0]
            
            
        return center_list,roi_binary_list,big_rec_list
    
    
    def load_params_from_folder(self,tradition_confit_folder_path:str):
        
        CHECK_INPUT_VALID(os.path.exists(tradition_confit_folder_path),True)
        CHECK_INPUT_VALID(os.listdir(tradition_confit_folder_path),['red','blue'])
        root_path = os.path.join(tradition_confit_folder_path,self.armor_color)
        preprocess_path = os.path.join(root_path,'preprocess_params.yaml')
        filter1_path = os.path.join(root_path,'filter1_params.yaml')
        filter2_path = os.path.join(root_path,'filter2_params.yaml')
        self.preprocess_bgr_params.load_params_from_yaml(preprocess_path)
        lr1.info(f'IMG : Load preprocess params success : {preprocess_path}')
        
        self.filter1.filter_params.load_params_from_yaml(filter1_path)
        lr1.info(f'IMG : Load filter1 params success : {filter1_path}')
        
        self.filter2.filter_params.load_params_from_yaml(filter2_path)
        lr1.info(f'IMG : Load filter2 params success : {filter2_path}')
        
                
    def enable_preprocess_config(self,window_name:str = 'preprocess_config',press_key_to_save:str = 's'):
        self.config_window_name =window_name
        self.press_key_to_save = press_key_to_save
        def for_trackbar(x):
            pass
        cv2.namedWindow(window_name,cv2.WINDOW_FREERATIO)
        cv2.createTrackbar('yuv_range_min',window_name,0,255,for_trackbar)
        cv2.createTrackbar('yuv_range_max',window_name,0,255,for_trackbar)
        cv2.createTrackbar('threshold',window_name,0,255,for_trackbar)
        
        if self.armor_color == 'red':
            cv2.setTrackbarPos('yuv_range_min',window_name,self.preprocess_bgr_params.red_armor_yuv_range[0])
            cv2.setTrackbarPos('yuv_range_max',window_name,self.preprocess_bgr_params.red_armor_yuv_range[1])
            cv2.setTrackbarPos('threshold',window_name,self.preprocess_bgr_params.red_armor_binary_roi_threshold)
        if self.armor_color == 'blue':
            cv2.setTrackbarPos('yuv_range_min',window_name,self.preprocess_bgr_params.blue_armor_yuv_range[0])
            cv2.setTrackbarPos('yuv_range_max',window_name,self.preprocess_bgr_params.blue_armor_yuv_range[1])
            cv2.setTrackbarPos('threshold',window_name,self.preprocess_bgr_params.blue_armor_binary_roi_threshold)
            
    
    def detect_trackbar_config(self):
        
        if self.armor_color ==  'red':
            
            self.preprocess_bgr_params.red_armor_yuv_range[0] = cv2.getTrackbarPos('yuv_range_min',self.config_window_name)
            self.preprocess_bgr_params.red_armor_yuv_range[1] = cv2.getTrackbarPos('yuv_range_max',self.config_window_name)
            self.preprocess_bgr_params.red_armor_binary_roi_threshold = cv2.getTrackbarPos('threshold',self.config_window_name)
        else:
            
            self.preprocess_bgr_params.blue_armor_yuv_range[0] = cv2.getTrackbarPos('yuv_range_min',self.config_window_name)
            self.preprocess_bgr_params.blue_armor_yuv_range[1] = cv2.getTrackbarPos('yuv_range_max',self.config_window_name)
            self.preprocess_bgr_params.blue_armor_binary_roi_threshold = cv2.getTrackbarPos('threshold',self.config_window_name)
            
        
        if cv2.waitKey(1) == ord(self.press_key_to_save):
            self.preprocess_bgr_params.save_params_to_yaml('preprocess_params.yaml')
        if cv2.waitKey(1) == ord(self.save_roi_key):
            if self.roi_single is not None:
                cv2.imwrite('roi_tmp.png',self.roi_single)
        
    
    
    
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
        
        big_rec_list = [make_big_rec(rec_pair[0],rec_pair[1]) for rec_pair in small_rec_pairs_list]\
                        if small_rec_pairs_list is not None else None 
        big_rec_list = expand_rec_wid(big_rec_list,EXPAND_RATE,img_size_yx=img_single.shape)
        
        big_rec_list = big_rec_list if not self.if_enable_filter2 \
                                else self.filter2.get_output(big_rec_list)
            
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
                               self.preprocess_bgr_params.gaussianblur_kernel_size,
                               self.preprocess_bgr_params.gaussianblur_x)
        y,u,v = cv2.split(out)
        
        if self.armor_color == 'red':
            out = cv2.inRange(v.reshape(img_bgr.shape[0],img_bgr.shape[1],1),
                              self.preprocess_bgr_params.red_armor_yuv_range[0],
                              self.preprocess_bgr_params.red_armor_yuv_range[1]
                              )
        else:
            out = cv2.inRange(u.reshape(img_bgr.shape[0],img_bgr.shape[1],1),
                              self.preprocess_bgr_params.blue_armor_yuv_range[0],
                              self.preprocess_bgr_params.blue_armor_yuv_range[1]
                              )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           self.preprocess_bgr_params.close_kernel_size)
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

            dst=gray_stretch(dst,self.preprocess_bgr_params.strech_max)

            if self.armor_color=='red':
                ret,dst=cv2.threshold(dst,self.preprocess_bgr_params.red_armor_binary_roi_threshold,255,cv2.THRESH_BINARY)
            else:    
                ret,dst=cv2.threshold(dst,self.preprocess_bgr_params.blue_armor_binary_roi_threshold,255,cv2.THRESH_BINARY)
                
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
        
    
        
        
        
        
        

class Filter_of_lightbar:
    def __init__(self,
                 mode:str = 'Dbg''Rel'
                 ) -> None:
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        
        self.filter_params = Filter_Lightbar_Params()
        self.mode = mode
    
    
    def get_output(self,input_list:list,img_bgr:Union[np.ndarray,None] = None)->Union[list,None]:
        
        """Get pairs of conts after filter

        Args:
            input_list (list): list of conts

        Returns:
            Union[list,None]: 
            if nothing remained , return None, else return [(cont1,cont2),...]
        """
        if input_list is None:
            return None
        
        tmp_list = []
        feature_tuple_list = []
        out_list = []
        
        if self.mode == 'Dbg':
            print(f'Filter Light Bar Begin : get conts {len(input_list)}')
        # One order filter
        for each_cont in input_list:
            x,y,wid,hei,rec_points_list,rec_area = getrec_info(each_cont)
            
            if hei == 0 :
                continue
            if wid == 0:
                wid = 1
            aspect = wid/hei
            if self.mode == 'Dbg':
                print('Light Bar Aspect : ',aspect)
                print('Light Bar Area : ',rec_area)
                if img_bgr is not None:
                    draw_single_cont(img_bgr,each_cont)
                    #add_text(img_bgr,'as',round(aspect,2),(round(x),round(y)),color=(255,255,255),scale_size=0.6)
                    #add_text(img_bgr,'ar',round(rec_area,2),(round(x),round(y+30)),color=(255,255,255),scale_size=0.6)
                    #add_text(img_bgr,'wi',round(wid,2),(round(x),round(y+60)),color=(255,255,255),scale_size=0.6)
                
            
            if  INRANGE(aspect,self.filter_params.accept_aspect_ratio_range) \
            and INRANGE(rec_area,self.filter_params.accept_area_range):
                tmp_list.append(each_cont)
        if self.mode == 'Dbg':
            print(f'Filter Light Bar after one order : {len(tmp_list)}')      
        
        # Two order filter
        for each_cont in tmp_list:
            
            x,y,wid,hei,rec_points_list,rec_area = getrec_info(each_cont)
            
            if hei == 0:
                aspect = 0
            else:
                aspect = wid/hei
            
            for feature_tuple in feature_tuple_list:
                two_area_ratio = rec_area/feature_tuple[0]
                two_aspect_ratio = aspect/feature_tuple[1]
                center_dis = CAL_EUCLIDEAN_DISTANCE((x,y),feature_tuple[2])
                
                if self.mode == "Dbg":
                    print("Light Bar Two Aspect Ratio : ", two_aspect_ratio)
                    print("Light Bar Two Area Ratio : ",two_area_ratio)
                    print("Light Bar Center Dis : ",center_dis)
                        
                if  INRANGE(two_area_ratio,self.filter_params.accept_two_area_ratio_range) \
                and INRANGE(two_aspect_ratio,self.filter_params.accept_two_aspect_ratio_range) \
                and INRANGE(center_dis,self.filter_params.accept_center_distance_range):
                    
                    out_list.append((feature_tuple[3] , each_cont))
            
            feature_tuple_list.append((rec_area,aspect,(x,y),each_cont))
        
        if self.mode == 'Dbg':
            print(f'Filter Light Bar after two order : {len(out_list)}')
        
        return None if len(out_list) == 0 else out_list
    
    def enable_trackbar_config(self,window_name:str = 'filter1_config',press_key_to_save:str = 's'):
        
        '''
        create trackbars for filter settings\n
        auto set default
        '''
        
        self.config_window_name =window_name
        self.press_key_to_save = press_key_to_save
        def for_trackbar(x):
            pass
        cv2.namedWindow(window_name,cv2.WINDOW_FREERATIO)
        #cv2.createTrackbar('area_range_min',window_name,1,5000,for_trackbar)
        #cv2.createTrackbar('area_range_max',window_name,1,5000,for_trackbar)
        cv2.createTrackbar('aspect_range_min_100',window_name,1,100,for_trackbar)       # 0.01 ~ 1 , cause wid/hei must smaller than 1 for lightbar
        cv2.createTrackbar('aspect_range_max_100',window_name,1,100,for_trackbar)
        cv2.createTrackbar('area_like_range_min_100',window_name,10,1000,for_trackbar)  #0.1 ~ 10 
        cv2.createTrackbar('area_like_range_max_100',window_name,10,1000,for_trackbar)
        cv2.createTrackbar('aspect_like_range_min_100',window_name,10,1000,for_trackbar)
        cv2.createTrackbar('aspect_like_range_max_100',window_name,10,1000,for_trackbar)
        #cv2.createTrackbar('center_dis_range_min',window_name,1,1000,for_trackbar)
        #cv2.createTrackbar('center_dis_range_max',window_name,1,1000,for_trackbar)
        
        
        #cv2.setTrackbarPos('area_range_min',window_name,self.filter_params.accept_area_range[0])
        #cv2.setTrackbarPos('area_range_max',window_name,self.filter_params.accept_area_range[1])
        cv2.setTrackbarPos('aspect_range_min_100',window_name,round(self.filter_params.accept_aspect_ratio_range[0] * 100))
        cv2.setTrackbarPos('aspect_range_max_100',window_name,round(self.filter_params.accept_aspect_ratio_range[1] * 100))
        cv2.setTrackbarPos('area_like_range_min_100',window_name,round(self.filter_params.accept_two_area_ratio_range[0] * 100))
        cv2.setTrackbarPos('area_like_range_max_100',window_name,round(self.filter_params.accept_two_area_ratio_range[1] * 100))
        cv2.setTrackbarPos('aspect_like_range_min_100',window_name,round(self.filter_params.accept_two_aspect_ratio_range[0] * 100))
        cv2.setTrackbarPos('aspect_like_range_max_100',window_name,round(self.filter_params.accept_two_aspect_ratio_range[1] * 100))
        #cv2.setTrackbarPos('center_dis_range_min',window_name,self.filter_params.accept_center_distance_range[0])
        #cv2.setTrackbarPos('center_dis_range_max',window_name,self.filter_params.accept_center_distance_range[1])

    
    def detect_trackbar_config(self):
        
        #self.filter_params.accept_area_range[0] = cv2.getTrackbarPos('area_range_min',self.config_window_name)
        #self.filter_params.accept_area_range[1] = cv2.getTrackbarPos('area_range_max',self.config_window_name)
        self.filter_params.accept_aspect_ratio_range[0] =  cv2.getTrackbarPos('aspect_range_min_100',self.config_window_name)/100
        self.filter_params.accept_aspect_ratio_range[1] =  cv2.getTrackbarPos('aspect_range_max_100',self.config_window_name)/100
        self.filter_params.accept_two_area_ratio_range[0] = cv2.getTrackbarPos('area_like_range_min_100',self.config_window_name)/100
        self.filter_params.accept_two_area_ratio_range[1] = cv2.getTrackbarPos('area_like_range_max_100',self.config_window_name)/100
        self.filter_params.accept_two_aspect_ratio_range[0] = cv2.getTrackbarPos('aspect_like_range_min_100',self.config_window_name)/100
        self.filter_params.accept_two_aspect_ratio_range[1] = cv2.getTrackbarPos('aspect_like_range_max_100',self.config_window_name)/100
        #self.filter_params.accept_center_distance_range[0] = cv2.getTrackbarPos('center_dis_range_min',self.config_window_name)
        #self.filter_params.accept_center_distance_range[1] = cv2.getTrackbarPos('center_dis_range_max',self.config_window_name)

        if cv2.waitKey(1) == ord(self.press_key_to_save):
            
            self.filter_params.save_params_to_yaml('./filter1_params.yaml')
        
        
    
    
    
            
            
            
class Filter_of_big_rec:
    
    def __init__(self,mode
                 ) -> None:
        
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        self.filter_params = Filter_Big_Rec_Params()
        self.mode = mode
        
    def get_output(self,input_list:list)->Union[list,None]:
        
        """Get pairs of conts after filter
        Args:
            input_list (list): list of conts

        Returns:
            Union[list,None]: 
            if nothing remained , return None, else return [(cont1,cont2),...]
        """
        if input_list is None:
            if self.mode == 'Dbg':
                print("Filter Big Rec Get Nothing")
            return None
        tmp_list = []
        
        if self.mode == 'Dbg':
            print(f'Filter Big Rec begin : get conts {len(input_list)}')
            
        # One order filter
        for each_cont in input_list:
            x,y,wid,hei,rec_points_list,rec_area = getrec_info(each_cont)
            
            
            if wid == 0 or hei == 0:
                continue
            
            aspect = wid/hei
            if self.mode == 'Dbg':
                
                print("Big Rec Aspect : ",aspect)
                print("Big Rec Area : ",rec_area)
            
            if  INRANGE(aspect,self.filter_params.accept_aspect_ratio_range) \
            and INRANGE(rec_area,self.filter_params.accept_area_range):
                tmp_list.append(each_cont)
              
            
        if self.mode == 'Dbg':
            print(f'Filter Big Rec after one order : {len(tmp_list)}')      
        
        return tmp_list


    def enable_trackbar_config(self,window_name:str = 'filter2_config',press_key_to_save:str = 's'):
    
        '''
        create trackbars for filter settings\n
        auto set default
        '''
        
        self.config_window_name =window_name
        self.press_key_to_save = press_key_to_save
        def for_trackbar(x):
            pass
        cv2.namedWindow(window_name,cv2.WINDOW_FREERATIO)
        #cv2.createTrackbar('area_range_min',window_name,1,10000,for_trackbar)
        #cv2.createTrackbar('area_range_max',window_name,1,10000,for_trackbar)
        cv2.createTrackbar('aspect_range_min_100',window_name,100,1000,for_trackbar)       # 1~10 , cause beg_rec wid/hei must bigger than 1 
        cv2.createTrackbar('aspect_range_max_100',window_name,100,1000,for_trackbar)

        
        #cv2.setTrackbarPos('area_range_min',window_name,self.filter_params.accept_area_range[0])
        #cv2.setTrackbarPos('area_range_max',window_name,self.filter_params.accept_area_range[1])
        cv2.setTrackbarPos('aspect_range_min_100',window_name,round(self.filter_params.accept_aspect_ratio_range[0] * 100))
        cv2.setTrackbarPos('aspect_range_max_100',window_name,round(self.filter_params.accept_aspect_ratio_range[1] * 100))


    def detect_trackbar_config(self):
        
        #self.filter_params.accept_area_range[0] = cv2.getTrackbarPos('area_range_min',self.config_window_name)
        #self.filter_params.accept_area_range[1] = cv2.getTrackbarPos('area_range_max',self.config_window_name)
        self.filter_params.accept_aspect_ratio_range[0] =  cv2.getTrackbarPos('aspect_range_min_100',self.config_window_name)/100
        self.filter_params.accept_aspect_ratio_range[1] =  cv2.getTrackbarPos('aspect_range_max_100',self.config_window_name)/100

        if cv2.waitKey(1) == ord(self.press_key_to_save):
            
            self.filter_params.save_params_to_yaml('./filter2_params.yaml')
        
        



############################################# Net Detector########################################################3


class Net_Detector:
    def __init__(self,
                 model_path:str,
                 class_yaml_path:str,
                 mode:str = 'Dbg',
                 engine_type:str = 'ort',
                 onnx_input_name:str = 'inputs',
                 onnx_output_name:str = 'outputs',
                 dtype:type = np.float32
                 ) -> None:
        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        CHECK_INPUT_VALID(engine_type,'ort','trt')
        
        self.dtype = dtype
        self.mode = mode
        self.engine_type = engine_type
        self.class_info = Data.get_file_info_from_yaml(class_yaml_path)
        if self.engine_type == 'ort':
            self.engine = Onnx_Engine(model_path,if_offline=True)
            self.onnx_inputname = onnx_input_name
            self.onnx_outputname = onnx_output_name



    @timing(1)
    def get_output(self,
                   input_list:Union[list,None]
                   )->Union[list,None]:
        """@timing

        Input:
            [roi_single1,roi_single2,...]

        Returns:
            [probability1,probability2,...],[armor_type1,armortype2,...]
        """
        if input_list is None:
            return None,None
        if len(input_list) is None:
            return None,None
        
        if self.engine_type == 'ort':
            inp = nomalize_for_onnx(input_list,dtype=self.dtype)
            output,ref_time =self.engine.run(output_nodes_name_list=None,
                            input_nodes_name_to_npvalue={self.onnx_inputname:inp})
            probabilities_list,index_list = trans_logits_in_batch_to_result(output[0])
            result_list = [self.class_info[index] for index in index_list]
            if self.mode == 'Dbg':
                print(f'Refence Time: {ref_time:.5f}')
                
            return probabilities_list,result_list
        
        
    

