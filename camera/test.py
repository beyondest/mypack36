import sys
sys.path.append('..')
import camera.mvsdk as mvsdk
from typing import Union,Optional
from logger.global_logger import lr1
from utils_network.data import *
import cv2
from os_op.basic import *
from mv_const import *


class Isp_Params:
    def __init__(self) -> None:
        self.exposure_time_us = 10 *1000
        self.gamma = 30
        
        
        gain = (100,100,100)
        self.r_gain,self.g_gain, self.b_gain = gain
        
        self.analog_gain = 64
        self.sharpness = 0
        self.saturation = 100
        self.contrast = 100
        
        
        self.resolution_wid = 1280
        self.resolution_hei = 1024
        

        if self.__len__ != len(ISP_PARAMS_SCOPE_LIST):
            lr1.critical(f"CAMERA : camera params length not match CAMERA_PAEAMS_SCOPE_LIST , {self.__len__} = {len(ISP_PARAMS_SCOPE_LIST)}")

        
        
    def print_show_all(self):
        for key, value in vars(self).items():
            print(f"{key} : {value}")
            
    def __len__(self):
        return len(vars(self))

    def load_params_from_yaml(self,yaml_path:str):
        
        reflect_dict =vars(self)
        setted_list = []
        info = Data.get_file_info_from_yaml(yaml_path)
        
        if len(info) != len(reflect_dict) :
            lr1.error(f"CAMERA : {yaml_path} has wrong params length {len(info)} , expected {len(reflect_dict)}")
            
        for i,item in enumerate(info.items()):
            key,value = item
            if key not in reflect_dict:
                lr1.error(f"CAMERA : camera params {key} : {value} from {yaml_path} failed, no such key")
            elif key in setted_list:
                lr1.error(f"CAMERA : camera params {key} dulplicated")
            else:
                value = CLAMP(value,ISP_PARAMS_SCOPE_LIST[i],True)
                reflect_dict[key] = value
                
            setted_list.append(key)
        
        
            
                
            
        


class Mindvision_Camera:
    
    def __init__(self,
                 device_id:int = 0,
                 device_nickname:Union[str,None] = None,
                 output_format:str = "RGB8",
                 if_auto_exposure:bool = False,
                 if_trigger_by_software:bool = False
                 ) -> None:
        
        self.isp_params = Isp_Params()
        self.config_params = Isp_Params()
        self.hcamera = self._init(device_id=device_id,
                                  device_nickname=device_nickname,
                                  output_format = output_format,
                                  if_auto_exposure = if_auto_exposure,
                                  if_trigger_by_software = if_trigger_by_software)

        
        
            
    def enable_trackbar_config(self,window_name:str):
        self.isp_window_name = window_name
        self._visualize_isp_config()        
    

    
    def start(self):
        pass
    
    
    
    def end(self):
        pass
    
    
    def isp_config(self,
                   exposure_time_us:Union[int,None] = None,
                   gamma:Union[int,None] = None,
                   gain:Union[tuple,None] = None,
                   analog_gain:Union[int,None] = None,
                   sharpness:Union[int,None] = None,
                   saturation:Union[int,None] = None,
                   contrast:Union[int,None] = None,
                   resolution_xy:Union[tuple,None] = None,
                   
                   ):
        
        if exposure_time_us is not None:
            exposure_time_us = CLAMP(exposure_time_us,ISP_PARAMS_SCOPE_LIST[0],True)
            mvsdk.CameraSetExposureTime(self.hcamera,exposure_time_us)
            
        if gamma is not None:
            gamma = CLAMP(gamma,ISP_PARAMS_SCOPE_LIST[1],True)
            mvsdk.CameraSetGamma(self.hcamera,gamma)
            
        if gain is not None:
            gain[0] = CLAMP(gain[0],ISP_PARAMS_SCOPE_LIST[2],True)
            gain[1] = CLAMP(gain[1],ISP_PARAMS_SCOPE_LIST[3],True)
            gain[2] = CLAMP(gain[2],ISP_PARAMS_SCOPE_LIST[4],True)
            mvsdk.CameraSetGain(self.hcamera,gain[0],gain[1],gain[2])
            
        if analog_gain is not None:
            analog_gain = CLAMP(analog_gain,ISP_PARAMS_SCOPE_LIST[5],True)
            mvsdk.CameraSetAnalogGain(self.hcamera,analog_gain)
            
        if sharpness is not None:
            sharpness = CLAMP(sharpness,ISP_PARAMS_SCOPE_LIST[6],True)
            mvsdk.CameraSetSharpness(self.hcamera,self.config_params.sharpness)
            
        if saturation is not None: 
            saturation = CLAMP(saturation,ISP_PARAMS_SCOPE_LIST[7],True)       
            mvsdk.CameraSetSaturation(self.hcamera,saturation)
            
        if contrast is not None:
            contrast = CLAMP(contrast,ISP_PARAMS_SCOPE_LIST[8],True)
            mvsdk.CameraSetContrast(self.hcamera,self.config_params.contrast)

        if resolution_xy is not None:
            
            resolution_xy[0] = CLAMP(resolution_xy[0],ISP_PARAMS_SCOPE_LIST[9],True)
            resolution_xy[1] = CLAMP(resolution_xy[1],ISP_PARAMS_SCOPE_LIST[10],True)
            mvsdk.CameraSetImageResolutionEx(self.hcamera,
                                             0xff,
                                             0,
                                             0,
                                             0,
                                             0,
                                             CAMERA_RESOLUTION_DEFAULT_XY[0],
                                             CAMERA_RESOLUTION_DEFAULT_XY[1],
                                             resolution_xy[0],
                                             resolution_xy[1]
                                             )
            
    
    def change_roi(self,
                   x,
                   y,
                   wid,
                   hei,
                   final_wid,
                   final_hei):
        
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
        

    
    def _init(self,
              device_id,
              device_nickname:Union[str,None] = None,
              output_format:str = 'RGB',
              if_auto_exposure:bool = False,
              if_trigger_by_software:bool = False):
        
        if device_id is None:
            lr1.critical("CAMERA: device_id cannot be None")
            
        dev_info_list = mvsdk.CameraEnumerateDevice()
        if len(dev_info_list) == 0:
            lr1.critical("CAMERA: no camera found")
        if len(dev_info_list) >1:
            lr1.warning(f"CAMERA: more than 1, {len(dev_info_list)} cameras found")
        
        if device_nickname is not None:
            for i in range(len(dev_info_list)):
                hcamera = mvsdk.CameraInit(dev_info_list[i])
                if mvsdk.CameraGetFriendlyName(hcamera) == device_nickname:
                    lr1.info(f"CAMERA: found {device_nickname} camera, id is {i}")
                    break
            else:
                lr1.error(f"CAMERA: {device_nickname} camera not found")
                raise TypeError("Not found specified camera")
        
        elif device_id+1 > len(dev_info_list):
            lr1.critical(f"CAMERA: id {device_id} out of range {len(dev_info_list)}")
        
        else:
            hcamera = mvsdk.CameraInit(dev_info_list[device_id])
        
        
        
        mvsdk.CameraSetIspOutFormat(hcamera,CAMERA_SHOW_TO_TYPE_DICT[output_format])
        
        mvsdk.CameraSetAeState(hcamera,if_auto_exposure)   # enable adjusting exposuretime manually
        
        mvsdk.CameraSetTriggerMode(hcamera,if_trigger_by_software)   # 0 means continous grab mode; 1 means soft trigger mode
        
        
        return hcamera
            
    def _get_camera_present_param(self):
        
        self.isp_params.exposure_time_us = mvsdk.CameraGetExposureTime(self.hcamera)
        self.isp_params.gamma = mvsdk.CameraGetGamma(self.hcamera)
        
        gain = mvsdk.CameraGetGain(self.hcamera)
        self.isp_params.r_gain,self.isp_params.g_gain,self.isp_params.b_gain = gain
        
        self.isp_params.analog_gain = mvsdk.CameraGetAnalogGain(self.hcamera)
        self.isp_params.sharpness = mvsdk.CameraGetSharpness(self.hcamera)
        self.isp_params.saturation = mvsdk.CameraGetSaturation(self.hcamera)
        self.isp_params.contrast = mvsdk.CameraGetContrast(self.hcamera)
        
        resolution = mvsdk.CameraGetImageResolution(self.hcamera)
        self.isp_params.resolution_wid = resolution.iWidth
        self.isp_params.resolution_hei = resolution.iHeight
        

        
    def _visualize_isp_config(self):
        self.trackbar_callback_list = []
        for i in self.trackbar_callback_list:
            self.trackbar_callback_list.append(self.isp_config())
        def _trackbar_callback(value):
            self._update_camera_param_from_trackbar()
            
            
        for index,name in enumerate(vars(self.isp_params).keys() ):
            cv2.createTrackbar(name ,
                               self.isp_window_name,
                               ISP_PARAMS_SCOPE_LIST[index][0],
                               ISP_PARAMS_SCOPE_LIST[index][1],
                               _trackbar_callback)
        self._get_camera_present_param()
        self._set_trackbar_pos_by_current_params()
        
    def _update_camera_param_from_trackbar(self):
        
        reflect_dict = vars(self.config_params)
        for key in reflect_dict.keys():
            reflect_dict[key] = cv2.getTrackbarPos(key,self.isp_window_name)


    def _set_trackbar_pos_by_current_params(self):
        
        reflect_dict = vars(self.isp_params)
        for key in reflect_dict.keys():
            cv2.setTrackbarPos(key,self.isp_window_name,reflect_dict[key])
    

    
        
                