import sys
sys.path.append('..')
from ..camera import mvsdk
from ..camera.mv_const import *
from ..os_op.basic import *
from ..os_op.global_logger import *
from ..utils_network import data

import yaml
import time
import numpy as np
import cv2
from typing import Union,Optional



class Isp_Params(Params):
    def __init__(self) -> None:
        super().__init__()
        self.exposure_time_us = CAMERA_EXPOSURE_TIME_US_DEFALUT
        self.gamma = CAMERA_GAMMA_DEFAULT
        self.r_gain,self.g_gain, self.b_gain = CAMERA_GAIN_DEFAULT
        
        self.analog_gain = CAMERA_ANALOG_GAIN_DEFAULT
        self.analog_gain_x = CAMERA_ANALOG_GAIN_X_DEFAULT
        self.sharpness = CAMERA_SHARPNESS_DEFAULT
        self.saturation = CAMERA_SATURATION_DEFAULT
        self.contrast = CAMERA_CONTRAST_DEFAULT
        self.grab_resolution_wid = CAMERA_GRAB_RESOLUTION_XY_DEFAUT[0] 
        self.grab_resolution_hei = CAMERA_GRAB_RESOLUTION_XY_DEFAUT[1]
        self.fps = CAMERA_FPS_DEFAULT
        


        if len(self) != len(ISP_PARAMS_SCOPE_LIST):
            lr1.critical(f"CAMERA : camera params length not match CAMERA_PARAMS_SCOPE_LIST , {len(self)} = {len(ISP_PARAMS_SCOPE_LIST)}")

        
        

            
    

        
            
                
            
        


class Mindvision_Camera(Custom_Context_Obj):
    
    def __init__(self,
                 device_id:int = 0,
                 device_nickname:Union[str,None] = None,
                 output_format:str = CAMERA_OUTPUT_DEFAULT,
                 if_auto_exposure:bool = False,
                 if_trigger_by_software:bool = False,
                 camera_run_platform:str = 'linux',
                 if_use_last_params:bool = False,
                 pingpong_exposure:Union[None,list] = None,
                 camera_mode:str = 'Dbg''Rel',
                 camera_config_folder:str = './camera_configs',
                 armor_color:str = 'red',
                 if_yolov5:bool = True
                 ) -> None:
        
        CHECK_INPUT_VALID(camera_run_platform,'linux','windows')
        CHECK_INPUT_VALID(camera_mode,'Dbg','Rel')
        CHECK_INPUT_VALID(armor_color,'red','blue')
        self.if_yolov5 = if_yolov5
        
        self.isp_params = Isp_Params()

        self.device_id = device_id
        self.device_nickname = device_nickname
        
        self.output_format = output_format
        self.if_trigger_by_software = if_trigger_by_software if pingpong_exposure is None else True
        self.if_auto_exposure = if_auto_exposure
        self.pingpong_exposure = pingpong_exposure
        self.if_use_last_params = if_use_last_params
        self.camera_mode = camera_mode
        self.roi_resolution_xy = CAMERA_ROI_RESOLUTION_XY_DEFAULT
        self.camera_run_platform = camera_run_platform
        self.pingpong_count = 0
        self.armor_color = armor_color
        self.random_config_count = 0
        self.if_enable_save_video = False
        self.if_save_img = False
        self.if_enable_trackbar_config = False
        self.hcamera = self._init()
        
        if not self.if_use_last_params:
            if camera_config_folder is not None:
                self.load_params_from_folder(camera_config_folder)
                lr1.info(f'CAMERA : Load params success from {camera_config_folder}')
                
            else:
                lr1.warning(f'CAMERA : Will not use yaml params nor last params, but use default params')
                
            self._isp_config_by_isp_params()
            self.change_roi(self.roi_resolution_xy[0],self.roi_resolution_xy[1])   
        else:
            lr1.info(f'CAMERA : Use Last params')
        
        self._update_camera_isp_params()
        
       
    
            
    def enable_trackbar_config(self,window_name:str = 'isp_config',press_key_to_save:str = 's',save_yaml_path:str = './tmp_isp_params.yaml'):
        
        self.isp_window_name = window_name
        self.press_key_to_save = press_key_to_save
        self.save_yaml_path = save_yaml_path
        self.if_enable_trackbar_config = True
        self._visualize_isp_config_by_isp_params()        
        
    def enable_save_img(self,
                        save_folder:str = './tmp_imgs',
                        img_name_suffix:str = 'jpg',
                        save_img_interval:Union[int,None] = 10,
                        press_key_to_save:Union[str,None] = None
                        ):
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder
        self.img_name_suffix = img_name_suffix
        self.if_save_img = True
        self.save_img_count = 0
        
        if save_img_interval is not None and press_key_to_save is not None:
            lr1.warning(f'CAMERA : save img both by interval and key press')
            
        self.save_img_interval = save_img_interval
        self.img_press_key_to_save = press_key_to_save
    
    
    def enbable_save_video(self,save_video_path:str = './tmp_video.avi',fps:int = 30):
        
        self.if_enable_save_video = True
        self.video_writer = cv2.VideoWriter()
        fourcc = self.video_writer.fourcc(*'mp4v')
        self.video_writer.open(save_video_path,fourcc,fps,(self.isp_params.grab_resolution_wid,self.isp_params.grab_resolution_hei))
    
    
    def load_params_from_folder(self,folder_path:str):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f'CAMERA : {folder_path} not exists')
        if self.if_yolov5:
            custom_isp_yaml_path = os.path.join(folder_path,'yolov5_isp_params.yaml')
        else:
            if self.armor_color == 'blue':
                custom_isp_yaml_path = os.path.join(folder_path,'blue_isp_params.yaml')
            else:
                custom_isp_yaml_path = os.path.join(folder_path,'red_isp_params.yaml')
                
        if not os.path.exists(custom_isp_yaml_path):
            raise FileNotFoundError(f'CAMERA : {custom_isp_yaml_path} not exists')
        
        self.isp_params.load_params_from_yaml(custom_isp_yaml_path)
        
        self._isp_config_by_isp_params()
    
    def save_all_params_to_file(self,file_name:str):
        mvsdk.CameraSaveParameterToFile(self.hcamera,file_name)
    
    def save_custom_params_to_yaml(self,yaml_path:str,mode:str = 'w'):
        self.isp_params.save_params_to_yaml(yaml_path,mode)
    
    def print_show_params(self):
        self.isp_params.print_show_all()
        print(f"roi resolution: {self.roi_resolution_xy}")
        print(f'device id: {self.device_id}')
        print(f'device nickname: {self.device_nickname}')
        print(f'output_format: {self.output_format}')
        trigger_mode = 'software trigger' if self.if_trigger_by_software else 'continous'
        print(f'trigger mode: {trigger_mode}')     
    

    def get_img(self)->Union[np.ndarray,list]:
        """
        Pingpong exposure: odd for exposure[0] , even for exposure[1]

        Returns:
            dst or [dst , pingpong count]
        """
        
        
        t1 = time.perf_counter()
        if self.if_enable_trackbar_config:
            self._detect_trackbar_config()
            
            
        prawdata,pframehead=mvsdk.CameraGetImageBuffer(self.hcamera,CAMERA_GRAB_IMG_WATI_TIME_MS)
        t2 = time.perf_counter()
        if self.pingpong_exposure is not None:
            self.pingpong_count +=1
            if self.pingpong_count == 10:
                self.pingpong_count = 0
            if (self.pingpong_count+1 )% 2:
                mvsdk.CameraSetExposureTime(self.hcamera,self.pingpong_exposure[0])
            else:
                mvsdk.CameraSetExposureTime(self.hcamera,self.pingpong_exposure[1])
        if self.if_trigger_by_software:
            print('trigger')
            mvsdk.CameraSoftTrigger(self.hcamera)
            
        mvsdk.CameraImageProcess(self.hcamera,prawdata,self.pframebuffer_address,pframehead)
        t3 = time.perf_counter()
        
        mvsdk.CameraReleaseImageBuffer(self.hcamera,prawdata)
        mvsdk.CameraFlipFrameBuffer(self.pframebuffer_address,pframehead,self.camera_run_platform=='windows')
        frame_data = (mvsdk.c_ubyte * pframehead.uBytes).from_address(self.pframebuffer_address)        # make an array with size ubyte (8) and length uBytes
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        dst=frame.reshape((self.roi_resolution_xy[1],self.roi_resolution_xy[0],CAMERA_CHANNEL_NUMS))
        dst = cv2.resize(dst,(self.isp_params.grab_resolution_wid,self.isp_params.grab_resolution_hei))
        
        if self.camera_mode == 'Dbg':
            lr1.debug(f'Camera : get buffer: { t2-t1:.4f}, camera isp process: { t3-t2:.4f}')
            
        
        if dst is not None and self.if_save_img:
            if self.save_img_interval is not None:
                self.save_img_count += 1
                if self.save_img_count % self.save_img_interval == 0:
                    img_name = f'{self.save_folder}/{time.time()}.{self.img_name_suffix}'
                    cv2.imwrite(img_name,dst)
                    
            if self.img_press_key_to_save is not None:
                if cv2.waitKey(1) & 0xFF == ord(self.img_press_key_to_save):
                    img_name = f'{self.save_folder}/{time.time()}.{self.img_name_suffix}'
                    cv2.imwrite(img_name,dst)
                    
        if dst is not None and self.if_enable_save_video:
            self.video_writer.write(dst)
            
        if self.pingpong_exposure is not None:
            return dst,self.pingpong_count
        else:
            return dst

    def trigger(self):
        if self.if_trigger_by_software:
            mvsdk.CameraSoftTrigger(self.hcamera)
        else:
            lr1.warning('CAMERA : trigger only works when if_trigger_by_software is True')



    def isp_config(self,
                   exposure_time_us:Union[int,None] = None,
                   gamma:Union[int,None] = None,
                   gain:Union[list,None] = None,
                   analog_gain:Union[int,None] = None,
                   analog_gain_x:Union[int,None] = None,
                   sharpness:Union[int,None] = None,
                   saturation:Union[int,None] = None,
                   contrast:Union[int,None] = None,
                   grab_resolution_wid:Union[int,None] = None,
                   grab_resolution_hei:Union[int,None] = None,
                   fps:Union[int,None] = None
                   ):
        
        if exposure_time_us is not None:
            exposure_time_us = CLAMP(exposure_time_us,ISP_PARAMS_SCOPE_LIST[0],True)
            self.isp_params.exposure_time_us = exposure_time_us
            mvsdk.CameraSetExposureTime(self.hcamera,exposure_time_us)
            
        if gamma is not None:
            gamma = CLAMP(gamma,ISP_PARAMS_SCOPE_LIST[1],True)
            self.isp_params.gamma = gamma
            mvsdk.CameraSetGamma(self.hcamera,gamma)
            
        if gain is not None:
            gain[0] = CLAMP(gain[0],ISP_PARAMS_SCOPE_LIST[2],True)
            gain[1] = CLAMP(gain[1],ISP_PARAMS_SCOPE_LIST[3],True)
            gain[2] = CLAMP(gain[2],ISP_PARAMS_SCOPE_LIST[4],True)
            self.isp_params.r_gain,self.isp_params.g_gain,self.isp_params.b_gain = gain
            mvsdk.CameraSetGain(self.hcamera,gain[0],gain[1],gain[2])
            
        if analog_gain is not None:
            analog_gain = CLAMP(analog_gain,ISP_PARAMS_SCOPE_LIST[5],True)
            self.isp_params.analog_gain = analog_gain
            mvsdk.CameraSetAnalogGain(self.hcamera,analog_gain)
        
        if analog_gain_x is not None:
            analog_gain_x = CLAMP(analog_gain_x,ISP_PARAMS_SCOPE_LIST[6],True)
            self.isp_params.analog_gain_x = analog_gain_x
            mvsdk.CameraSetAnalogGainX(self.hcamera,analog_gain_x)
            
        if sharpness is not None:
            sharpness = CLAMP(sharpness,ISP_PARAMS_SCOPE_LIST[7],True)
            self.isp_params.sharpness = sharpness
            mvsdk.CameraSetSharpness(self.hcamera,self.isp_params.sharpness)
            
        if saturation is not None: 
            saturation = CLAMP(saturation,ISP_PARAMS_SCOPE_LIST[8],True)   
            self.isp_params.saturation = saturation    
            mvsdk.CameraSetSaturation(self.hcamera,saturation)
            
        if contrast is not None:
            contrast = CLAMP(contrast,ISP_PARAMS_SCOPE_LIST[9],True)
            self.isp_params.contrast = contrast
            mvsdk.CameraSetContrast(self.hcamera,self.isp_params.contrast)

        if grab_resolution_wid is not None:
            grab_resolution_wid  = CLAMP(grab_resolution_wid,ISP_PARAMS_SCOPE_LIST[10],True)
            self.isp_params.grab_resolution_wid = grab_resolution_wid
            
        if grab_resolution_hei is not None:
            grab_resolution_hei = CLAMP(grab_resolution_hei,ISP_PARAMS_SCOPE_LIST[11],True)
            self.isp_params.grab_resolution_hei = grab_resolution_hei

        if fps is not None:
            fps = CLAMP(fps,ISP_PARAMS_SCOPE_LIST[12],True)
            self.fps = fps
            mvsdk.CameraSetFrameSpeed(self.hcamera,self.fps)
            
        

    
    def change_roi(self,
                   wid,
                   hei,
                   x = 0,
                   y = 0,
                   final_wid = 0,
                   final_hei = 0):
        
        mvsdk.CameraSetImageResolutionEx(self.hcamera,
                                         0xff,
                                         0,
                                         0,
                                         x,
                                         y,
                                         wid,
                                         hei,
                                         final_wid,
                                         final_hei)
        self.roi_resolution_xy = [wid,hei]
        
        
    def _detect_trackbar_config(self):
        
        self._update_camera_isp_params_from_trackbar()
        self._isp_config_by_isp_params()
        
        if cv2.waitKey(1) & 0xff == ord(self.press_key_to_save):
            self.save_custom_params_to_yaml(self.save_yaml_path)
    

    def random_config(self,interval:int = 100):
        self.random_config_count += 1
        if self.random_config_count % interval == 0:
            
            self.isp_params.exposure_time_us = np.random.randint(ISP_PARAMS_SCOPE_LIST[0][0],ISP_PARAMS_SCOPE_LIST[0][1])
            self._isp_config_by_isp_params()
        
    def _init(self):
        
        if self.device_id is None:
            lr1.critical("CAMERA: device_id cannot be None")
            
        dev_info_list = mvsdk.CameraEnumerateDevice()
        if len(dev_info_list) == 0:
            lr1.critical("CAMERA: no camera found")
            raise TypeError("no camera found")
        if len(dev_info_list) >1:
            lr1.warning(f"CAMERA: more than 1, {len(dev_info_list)} cameras found")
        
        if self.device_nickname is not None:
            for i in range(len(dev_info_list)):
                hcamera = mvsdk.CameraInit(dev_info_list[i])
                if mvsdk.CameraGetFriendlyName(hcamera) == self.device_nickname:
                    lr1.info(f"CAMERA: found {self.device_nickname} camera, id is {i}")
                    break
            else:
                lr1.error(f"CAMERA: {self.device_nickname} camera not found")
                raise TypeError("Not found specified camera")
        
        elif self.device_id+1 > len(dev_info_list):
            lr1.critical(f"CAMERA: id {self.device_id} out of range {len(dev_info_list)}")
        
        else:
            hcamera = mvsdk.CameraInit(dev_info_list[self.device_id])
            
        if not self.if_use_last_params:
            mvsdk.CameraSetIspOutFormat(hcamera,CAMERA_SHOW_TO_TYPE_DICT[self.output_format])
            mvsdk.CameraSetAeState(hcamera,self.if_auto_exposure)   # enable adjusting exposuretime manually
            mvsdk.CameraSetTriggerMode(hcamera,self.if_trigger_by_software)   # 0 means continous grab mode; 1 means soft trigger mode
            
        lr1.info(f'CAMERA : CAMERA id {self.device_id} init success')
        
        return hcamera
            
            
    def _update_camera_isp_params(self):
        
        self.isp_params.exposure_time_us =round(mvsdk.CameraGetExposureTime(self.hcamera))
        self.isp_params.gamma = mvsdk.CameraGetGamma(self.hcamera)
        
        gain = mvsdk.CameraGetGain(self.hcamera)
        self.isp_params.r_gain,self.isp_params.g_gain,self.isp_params.b_gain = gain
        
        self.isp_params.analog_gain = mvsdk.CameraGetAnalogGain(self.hcamera)
        self.isp_params.analog_gain_x = mvsdk.CameraGetAnalogGainX(self.hcamera)
        self.isp_params.sharpness = mvsdk.CameraGetSharpness(self.hcamera)
        self.isp_params.saturation = mvsdk.CameraGetSaturation(self.hcamera)
        self.isp_params.contrast = mvsdk.CameraGetContrast(self.hcamera)
        
        resolution = mvsdk.CameraGetImageResolution(self.hcamera)
        self.roi_resolution_xy[0] = resolution.iWidth 
        self.roi_resolution_xy[1] = resolution.iHeight
        self.fps = mvsdk.CameraGetFrameSpeed(self.hcamera)

         
    def _visualize_isp_config_by_isp_params(self):

        def for_trackbar(x):
            
            pass

        cv2.namedWindow(self.isp_window_name,cv2.WINDOW_FREERATIO)
        for index,name in enumerate(vars(self.isp_params).keys() ):
            cv2.createTrackbar(name ,
                               self.isp_window_name,
                               ISP_PARAMS_SCOPE_LIST[index][0],
                               ISP_PARAMS_SCOPE_LIST[index][1],
                               for_trackbar)
            
        self._update_camera_isp_params()
        self._set_trackbar_pos_by_present_params()
       
       
             
    def _set_trackbar_pos_by_present_params(self):
        
        reflect_dict = vars(self.isp_params)
        for key in reflect_dict.keys():
            tmp = round(reflect_dict[key])
            cv2.setTrackbarPos(key,self.isp_window_name,tmp)
    
    
    def _update_camera_isp_params_from_trackbar(self):
        
        
        reflect_dict = vars(self.isp_params)
        for key in reflect_dict.keys():
            reflect_dict[key] = cv2.getTrackbarPos(key,self.isp_window_name)
            
            
    def _isp_config_by_isp_params(self):
        
        
        mvsdk.CameraSetExposureTime(self.hcamera,self.isp_params.exposure_time_us)
        
        mvsdk.CameraSetGamma(self.hcamera,self.isp_params.gamma)
        
        mvsdk.CameraSetGain(self.hcamera,self.isp_params.r_gain,self.isp_params.g_gain,self.isp_params.b_gain)
        
        mvsdk.CameraSetAnalogGain(self.hcamera,self.isp_params.analog_gain)
        
        mvsdk.CameraSetAnalogGainX(self.hcamera,self.isp_params.analog_gain_x)
        
        mvsdk.CameraSetSharpness(self.hcamera,self.isp_params.sharpness)
        
        mvsdk.CameraSetSaturation(self.hcamera,self.isp_params.saturation)

        mvsdk.CameraSetContrast(self.hcamera,self.isp_params.contrast)

        #mvsdk.CameraSetFrameSpeed(self.hcamera,self.isp_params.fps)
        

        
        
        
    
    def _start(self):
        self.pframebuffer_address = mvsdk.CameraAlignMalloc(self.roi_resolution_xy[0] * self.roi_resolution_xy[1] * CAMERA_CHANNEL_NUMS,
                                                            CAMERA_ALIGN_BYTES_NUMS)
        mvsdk.CameraClearBuffer(self.hcamera)
        mvsdk.CameraPlay(self.hcamera)
        if self.if_trigger_by_software:
            mvsdk.CameraSoftTrigger(self.hcamera)
        lr1.info(f"CAMERA : camera id {self.device_id} , nickname {self.device_nickname} start play")
    
    
    def _end(self):
        
        """must be called 
        """
        
        mvsdk.CameraUnInit(self.hcamera)
        mvsdk.CameraAlignFree(self.pframebuffer_address)
        lr1.info(f"CAMERA : camera id {self.device_id} , nickname {self.device_nickname} closed")
        
        if self.if_enable_save_video:
            self.video_writer.release()
            lr1.info(f"CAMERA :  video released")
            
            
    def _errorhandler(self,exc_value):
        print(exc_value)
        print(type(exc_value))

 
        
    
    
                