      
from .const import *
from ..os_op.basic import *
from .tools import *
  
  
  
  
class Filter_Base:
    def __init__(self) -> None:
        pass
    
    def get_output(self):
        raise NotImplementedError('Please implement the get_output method')
    
    def enable_trackbar_config(self):
        raise NotImplementedError('Please implement the enable_trackbar_config method')
    
    def detect_trackbar_config(self):
        raise NotImplementedError('Please implement the detect_trackbar_config method')
    

    
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
        
        
        
    
        

class Filter_of_lightbar(Filter_Base):
    
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
                
                each_dict = {'cont':each_cont,'center':[x,y]}
                tmp_list.append(each_dict)
        
        sorted(tmp_list,key=lambda x:x['center'][0])
        tmp_list = [each_dict['cont'] for each_dict in tmp_list]
        
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
        
        
    
    
    
            
            
            
class Filter_of_big_rec(Filter_Base):
    
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
        
        

