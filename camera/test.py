import sys
sys.path.append('..')
import camera.mvsdk as mvsdk
from typing import Union,Optional
from logger.global_logger import lr1
from utils_network.data import *
from os_op.basic import *
from mv_const import *
import img.img_operation as imo

import cv2


def fuck(x):
    pass

class Isp_Params:
    def __init__(self) -> None:
        self.exposure_time_us = CAMERA_EXPOSURE_TIME_US_DEFALUT
        self.gamma = CAMERA_GAMMA_DEFAULT
        self.r_gain,self.g_gain, self.b_gain = CAMERA_GAIN_DEFAULT
        
        self.analog_gain = CAMERA_ANALOG_GAIN_DEFAULT
        self.sharpness = CAMERA_SHARPNESS_DEFAULT
        self.saturation = CAMERA_SATURATION_DEFAULT
        self.contrast = CAMERA_CONTRAST_DEFAULT
        
        
        self.resolution_wid = CAMERA_RESOLUTION_DEFAULT_XY[0]
        self.resolution_hei = CAMERA_RESOLUTION_DEFAULT_XY[1]
        

        if len(self) != len(ISP_PARAMS_SCOPE_LIST):
            lr1.critical(f"CAMERA : camera params length not match CAMERA_PARAMS_SCOPE_LIST , {len(self)} = {len(ISP_PARAMS_SCOPE_LIST)}")

        
        
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
                 if_trigger_by_software:bool = False,
                 camera_run_platform:str = 'linux',
                 if_show_init_params:bool = True
                 ) -> None:
        
        if camera_run_platform != 'linux' and camera_run_platform != 'windows':
            lr1.error(f'CAMERA: camera_run_platform {camera_run_platform} wrong, must be windows or linux')
            
        self.isp_params = Isp_Params()
        self.isp_params = Isp_Params()
        self.roi_resolution = CAMERA_RESOLUTION_DEFAULT_XY
        self.camera_run_platform = camera_run_platform
        self.if_trigger_by_software = if_trigger_by_software
        self.device_id = device_id
        self.device_nickname = device_nickname
        self.output_format = output_format
        self.if_auto_exposure = if_auto_exposure
        self.hcamera = self._init()


        
        if if_show_init_params:
            self.isp_params.print_show_all()
            print(f'device id: {self.device_id}')
            print(f'device nickname: {self.device_nickname}')
            print(f'output_format: {self.output_format}')
            trigger_mode = 'software trigger' if if_trigger_by_software else 'continous'
            print(f'trigger mode: {trigger_mode}')        
    
            
    def enable_trackbar_config(self,window_name:str):
        
        self.isp_window_name = window_name
        self._visualize_isp_config_by_trackbar()        
    
    def load_params_from_yaml(self,yaml_path:str):
        
        self.isp_params.load_params_from_yaml(yaml_path)
        self._isp_config_by_trackbar()
    
    
    
    def get_img_continous(self):
        
        prawdata,pframehead=mvsdk.CameraGetImageBuffer(self.hcamera,CAMERA_GRAB_IMG_WATI_TIME_MS)
        mvsdk.CameraImageProcess(self.hcamera,prawdata,self.pframebuffer_address,pframehead)
        mvsdk.CameraReleaseImageBuffer(self.hcamera,prawdata)
        
        mvsdk.CameraFlipFrameBuffer(self.pframebuffer_address,pframehead,self.camera_run_platform=='windows')
        frame_data = (mvsdk.c_ubyte * pframehead.uBytes).from_address(self.pframebuffer_address)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        dst=frame.reshape((self.isp_params.resolution_hei,self.isp_params.resolution_wid,CAMERA_CHANNEL_NUMS))
        
        return dst
        
    
    def get_img_softrigger(self)->Union[np.ndarray,None]:
        
        """get img in softtrigger mode, perhaps it help raise up FPS

        Returns:
            np.ndarray if SUCCESS else NONE
        """
        prawdata,pframehead =  mvsdk.CameraGetImageBuffer(self.hcamera,CAMERA_GRAB_IMG_WATI_TIME_MS)
        
        if prawdata is None:
            
            lr1.warning("CAMERA: No img in buffer, enable soft trigger")
            mvsdk.CameraSoftTrigger(self.hcamera)
            return None
            
        mvsdk.CameraSoftTrigger(self.hcamera) 
        mvsdk.CameraImageProcess(self.hcamera,
                                 pbyIn=prawdata,
                                 pbyOut=self.pframebuffer_address,
                                 pFrInfo=pframehead)
        mvsdk.CameraReleaseImageBuffer(self.hcamera,prawdata)
        
        mvsdk.CameraFlipFrameBuffer(self.pframebuffer_address,pframehead,self.camera_run_platform == 'windows')
        frame_data = (mvsdk.c_ubyte * pframehead.uBytes).from_address(self.pframebuffer_address)
        frame = np.frombuffer(frame_data,dtype=np.uint8)
        dst  = frame.reshape((self.isp_params.resolution_hei,self.isp_params.resolution_wid,CAMERA_CHANNEL_NUMS))
        
        return dst
    
        

    def isp_config(self,
                   exposure_time_us:Union[int,None] = None,
                   gamma:Union[int,None] = None,
                   gain:Union[list,None] = None,
                   analog_gain:Union[int,None] = None,
                   sharpness:Union[int,None] = None,
                   saturation:Union[int,None] = None,
                   contrast:Union[int,None] = None,
                   resolution_xy:Union[list,None] = None,
                   
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
            
        if sharpness is not None:
            sharpness = CLAMP(sharpness,ISP_PARAMS_SCOPE_LIST[6],True)
            self.isp_params.sharpness = sharpness
            mvsdk.CameraSetSharpness(self.hcamera,self.isp_params.sharpness)
            
        if saturation is not None: 
            saturation = CLAMP(saturation,ISP_PARAMS_SCOPE_LIST[7],True)   
            self.isp_params.saturation = saturation    
            mvsdk.CameraSetSaturation(self.hcamera,saturation)
            
        if contrast is not None:
            contrast = CLAMP(contrast,ISP_PARAMS_SCOPE_LIST[8],True)
            self.isp_params.contrast = contrast
            mvsdk.CameraSetContrast(self.hcamera,self.isp_params.contrast)

        if resolution_xy is not None:
            
            resolution_xy[0] = CLAMP(resolution_xy[0],ISP_PARAMS_SCOPE_LIST[9],True)
            resolution_xy[1] = CLAMP(resolution_xy[1],ISP_PARAMS_SCOPE_LIST[10],True)
            self.isp_params.resolution_wid = resolution_xy[0]
            self.isp_params.resolution_hei = resolution_xy[1]
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
        

    
    def _init(self):
        
        if self.device_id is None:
            lr1.critical("CAMERA: device_id cannot be None")
            
        dev_info_list = mvsdk.CameraEnumerateDevice()
        
        if len(dev_info_list) == 0:
            lr1.critical("CAMERA: no camera found")
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
        self.isp_params.sharpness = mvsdk.CameraGetSharpness(self.hcamera)
        self.isp_params.saturation = mvsdk.CameraGetSaturation(self.hcamera)
        self.isp_params.contrast = mvsdk.CameraGetContrast(self.hcamera)
        
        resolution = mvsdk.CameraGetImageResolution(self.hcamera)
        self.isp_params.resolution_wid = resolution.iWidth 
        self.isp_params.resolution_hei = resolution.iHeight

         
    def _visualize_isp_config_by_trackbar(self):
        self.ok = False
        def for_trackbar(x):
            
            if self.ok:
                self._update_camera_isp_params_from_trackbar()
                self._isp_config_by_trackbar()
        
        cv2.namedWindow(self.isp_window_name,cv2.WINDOW_FREERATIO)
        for index,name in enumerate(vars(self.isp_params).keys() ):
            cv2.createTrackbar(name ,
                               self.isp_window_name,
                               ISP_PARAMS_SCOPE_LIST[index][0],
                               ISP_PARAMS_SCOPE_LIST[index][1],
                               for_trackbar)
            
        self._update_camera_isp_params()
        self._set_trackbar_pos_by_present_params()
        self.ok =True
        
    def _set_trackbar_pos_by_present_params(self):
        
        reflect_dict = vars(self.isp_params)
        for key in reflect_dict.keys():
            cv2.setTrackbarPos(key,self.isp_window_name,reflect_dict[key])
    
    def _update_camera_isp_params_from_trackbar(self):
        
        """Only for trackbar callback
        """
        reflect_dict = vars(self.isp_params)
        for key in reflect_dict.keys():
            reflect_dict[key] = cv2.getTrackbarPos(key,self.isp_window_name)

    def _isp_config_by_trackbar(self):
        """Only for trackbar callback
        """
            
        mvsdk.CameraSetExposureTime(self.hcamera,self.isp_params.exposure_time_us)
        
        mvsdk.CameraSetGamma(self.hcamera,self.isp_params.gamma)
        
        mvsdk.CameraSetGain(self.hcamera,self.isp_params.r_gain,self.isp_params.g_gain,self.isp_params.b_gain)
        
        mvsdk.CameraSetAnalogGain(self.hcamera,self.isp_params.analog_gain)
        
        mvsdk.CameraSetSharpness(self.hcamera,self.isp_params.sharpness)
        
        mvsdk.CameraSetSaturation(self.hcamera,self.isp_params.saturation)

        mvsdk.CameraSetContrast(self.hcamera,self.isp_params.contrast)

        mvsdk.CameraSetImageResolutionEx(self.hcamera,
                                         0xff,
                                         0,
                                         0,
                                         0,
                                         0,
                                         self.isp_params.resolution_wid,
                                         self.isp_params.resolution_hei,
                                         0,
                                         0)
    
        
        
        
    
    def _start(self):
        self.pframebuffer_address = mvsdk.CameraAlignMalloc(self.roi_resolution[0] * self.roi_resolution[1] * CAMERA_CHANNEL_NUMS,
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
    
    def _errorhandler(self,exc_value):
        print(exc_value)
        print(type(exc_value))



if __name__ == "__main__":
    
    ca = Mindvision_Camera(if_trigger_by_software=False)
    fps = 0
    ca.enable_trackbar_config('config')
    with Context('camera',ca):
        while 1:
            t1 = time.perf_counter()
            

            img = ca.get_img_continous()
            
            imo.add_text(img,'fps',fps,scale_size=1)
            cv2.imshow('h',img)
            
            t2 = time.perf_counter()
            t = t2 - t1
            fps = round(1/t)
            
            if cv2.waitKey(1) ==27:
                break
        
    cv2.destroyAllWindows()
        
        
        
    
    
                