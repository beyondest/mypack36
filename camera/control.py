'''only consider one camera'''
'''need to config on spot'''

'''notice that exposure_time is us !!!'''

import sys
sys.path.append('..')
import camera.mvsdk as mvsdk
import cv2
import numpy as np
import platform
import os
from typing import Optional,Union
#gamma,exposure_time_us,rgb_gain,analog_gain,saturation,sharpness,contrast
isp_params_list=[]  
#for red
isp_params_list.append([16,1000,100,89,101,64,147,0,143])
#for blue
isp_params_list.append([26,1000,69,100,104,64,111,0,100])
#area_range,normal_ratio,strange_ratio1,strange_ratio2,shape_like_range,center_dis_range
#!!!NOTICE: some params need to multiply by 100!!!
filter_params_list=[]

#for blue and red
filter_params_list.append([(1, 277),156,1911,484,(13, 180),(2, 89)])



def camera_open(hcamera):
    '''
    let sdk work
    '''
    mvsdk.CameraPlay(hcamera)
    return 0


def camera_close(hcamera,pframebuffer_address):
    '''
    uninit the camera and release the buffer
    '''
    mvsdk.CameraUnInit(hcamera)
    mvsdk.CameraAlignFree(pframebuffer_address)


def camera_setframebuffer(size:int=1280*1024*3):
    pframebuffer_address=mvsdk.CameraAlignMalloc(size)
    return pframebuffer_address
    

def grab_img(
             hcamera,
             pframebuffer_address,
             wtime=1000
             )->np.ndarray:
    '''
    grab images as png\n
    interval: x frames\n
    '''
    
    
    
    prawdata,pframehead=mvsdk.CameraGetImageBuffer(hcamera,wtime)
    mvsdk.CameraImageProcess(hcamera,prawdata,pframebuffer_address,pframehead)
    mvsdk.CameraReleaseImageBuffer(hcamera,prawdata)
    if platform.system()   ==   'Windows':
        mvsdk.CameraFlipFrameBuffer(pframebuffer_address,pframehead,True)
    frame_data = (mvsdk.c_ubyte * pframehead.uBytes).from_address(pframebuffer_address)
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    dst=frame.reshape((1024,1280,3))
    return dst

def camera_init(out_put_format=mvsdk.CAMERA_MEDIA_TYPE_YUV8_UYV):
    '''
    NO.1:enumerate\n
    NO.2:set isp output format\n
    NO.3:set trigger mode\n
    Params
    @format: default=YUV8_UYV
    
    '''
    devinfo=mvsdk.CameraEnumerateDevice()
    hcamera=mvsdk.CameraInit(devinfo[0])
    mvsdk.CameraSetTriggerMode(hcamera,0)
    #24 bit
    mvsdk.CameraSetIspOutFormat(hcamera,out_put_format)
    
    
    return hcamera


def set_isp(hcamera,
             exposure_time_us=10*1000,
             gamma=30,
             r_gain=100,
             g_gain=100,
             b_gain=100,
             analog_gain=64,
             analog_gainx=2.0,
             sharpness=0,
             saturation=100,
             contrast=100,
             ):
    '''
    adjust images by 3 steps\n
    NO.1: brightness setting\n
    NO.2: color setting\n
    NO.3: contrast setting\n
    Params\n
    @exposure_time_us: exposure time default= 10*1000 us, range=(0, 31776.2, 3.9)
    @gamma: default=30, range(0,250)
    @rgb_gain: defalt=(100,100,100), range=(0,400)
    @gain_a: ??? defalt=64, range(64,192)
    @gainx_a: ??? defalt=2,0, range(2.0,6.0,0.03125)
    @sharpness: defalt=0, range(0,100)
    @saturation defalt=100, range(0,200)
    @contrast defalt=100, range(0,200)
    
    '''
    
    
    mvsdk.CameraSetAeState(hcamera,False)
    mvsdk.CameraSetExposureTime(hcamera,exposure_time_us)
    
    mvsdk.CameraSetGamma(hcamera,gamma)
    
    mvsdk.CameraSetGain(hcamera,r_gain,g_gain,b_gain)
    
    mvsdk.CameraSetAnalogGain(hcamera,analog_gain)
    
    mvsdk.CameraSetAnalogGainX(hcamera,analog_gainx)
    
    mvsdk.CameraSetSharpness(hcamera,sharpness)
    
    mvsdk.CameraSetSaturation(hcamera,saturation)

    mvsdk.CameraSetContrast(hcamera,contrast)
    
    
    
def lut_init(hcamera):
    '''emlutmode default=0 ???'''
    mvsdk.CameraSetLutMode(hcamera,emLutMode=0)
    #mvsdk.CameraSetCustomLut()
    
 
    
 

def camera_show(dst):
    '''
    visualize every seized frame in video
    '''
    cv2.imshow('fuck',dst)
    


def get_all(hcamera)->dict:
    '''get info of camera'''
    out={}
    
    cap=mvsdk.CameraGetCapability(hcamera)
    media_cap=mvsdk.CameraGetMediaType(hcamera)
    
    #out.update({'preset_lut':cap.iPresetLut})
    #out.update({'lut_mode':mvsdk.CameraGetLutMode(hcamera)})
    #out.update({'lut_preset_sel':mvsdk.CameraGetLutPresetSel(hcamera)})
    #out.update({'custom_lut':mvsdk.CameraGetCustomLut(hcamera)})
    #out.update({'contrast_range':cap.sContrastRange})
    out.update({'contrast':mvsdk.CameraGetContrast(hcamera)})
    #out.update({'exposure_time_range':mvsdk.CameraGetExposureTimeRange(hcamera)})
    out.update({'exposure_time_us':mvsdk.CameraGetExposureTime(hcamera)})
    #out.update({'gainxrange_a':mvsdk.CameraGetAnalogGainXRange(hcamera)})
    out.update({'gainx_a':mvsdk.CameraGetAnalogGainX(hcamera)})
    #out.update({'gainrange_ae_a':mvsdk.CameraGetAeAnalogGainRange(hcamera)})
    out.update({'gain_a':mvsdk.CameraGetAnalogGain(hcamera)})
    #out.update({'gain_range_rgb':cap.sRgbGainRange})
    out.update({'gain_rgb':mvsdk.CameraGetGain(hcamera)})
    #out.update({'gamma_range':cap.sGammaRange})
    out.update({'gamma':mvsdk.CameraGetGamma(hcamera)})
    #out.update({'saturation_range':cap.sSaturationRange})
    out.update({'saturation':mvsdk.CameraGetSaturation(hcamera)})
    #out.update({'sharpness_range':cap.sSharpnessRange})
    out.update({'sharpness':mvsdk.CameraGetSharpness(hcamera)})
    out.update({'trigger_mode':mvsdk.CameraGetTriggerMode(hcamera)})
    out.update({'isp_out_format':mvsdk.CameraGetIspOutFormat(hcamera)})
    out.update({'wb_mode':mvsdk.CameraGetWbMode(hcamera)})
    out.update({'black_level':mvsdk.CameraGetBlackLevel(hcamera)})
    out.update({'white_level':mvsdk.CameraGetWhiteLevel(hcamera)})
    #out.update({'out_format':media_cap.iMediatype})
    
    #out.update({'user_clr_temp_gain':mvsdk.CameraGetUserClrTempGain(hcamera)})
    #out.update({'user_clr_temp_matrix':mvsdk.CameraGetUserClrTempMatrix(hcamera)})

    #out.update({'isp_model_info':cap.sIspCapacity})
    #out.update({'dev_pre_info':cap.sCameraCapbility})
    '''
    out.update({'####################################':0})
    out.update({'clr_temp_mode':mvsdk.CameraGetClrTempMode(hcamera)})
    out.update({'current_param_groups':mvsdk.CameraGetCurrentParameterGroup(hcamera)})
    out.update({'denoise_3d_params':mvsdk.CameraGetDenoise3DParams(hcamera)})
    out.update({'eye_count':mvsdk.CameraGetEyeCount(hcamera)})
    out.update({'frame_id':mvsdk.CameraGetFrameID(hcamera)})
    out.update({'frame_speed':mvsdk.CameraGetFrameSpeed(hcamera)})
    out.update({'frame_statistic':mvsdk.CameraGetFrameStatistic(hcamera)})
    out.update({'frame_time_stamp':mvsdk.CameraGetFrameTimeStamp(hcamera)})
    out.update({'hdr':mvsdk.CameraGetHDR(hcamera)})
    out.update({'hdrgain_mode':mvsdk.CameraGetHDRGainMode(hcamera)})

    out.update({'media_type':mvsdk.CameraGetMediaType(hcamera)})
    out.update({'noise_filter_state':mvsdk.CameraGetNoiseFilterState(hcamera)})
    out.update({'parameter_mode':mvsdk.CameraGetParameterMode(hcamera)})
    out.update({'preset_clr_temp':mvsdk.CameraGetPresetClrTemp(hcamera)})
    out.update({'undistort_enable':mvsdk.CameraGetUndistortEnable(hcamera)})
    out.update({'undistort_params':mvsdk.CameraGetUndistortParams(hcamera)})
    '''


    
    
    
    
    return out



def print_getall(info:dict):
    print('*******************************************')
    for i in info:
        print(f'{i} : {info[i]}')
    print('*******************************************')

def camera_correct_white(hcamera,on_or_off)->int:
    '''return whitelevel'''
    if on_or_off == 1:
        mvsdk.CameraSetWbMode(hcamera,False)
        mvsdk.CameraSetOnceWB(hcamera)
    return mvsdk.CameraGetWhiteLevel(hcamera)


def camera_correct_black(hcamera,on_or_off)->int:
    '''return blacklevel'''
    if on_or_off == 1:
        mvsdk.CameraSetOnceBB(hcamera)
    return mvsdk.CameraGetBlackLevel(hcamera)


def PrintCapbility(cap):
	for i in range(cap.iTriggerDesc):
		desc = cap.pTriggerDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iImageSizeDesc):
		desc = cap.pImageSizeDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iClrTempDesc):
		desc = cap.pClrTempDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iMediaTypeDesc):
		desc = cap.pMediaTypeDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iFrameSpeedDesc):
		desc = cap.pFrameSpeedDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iPackLenDesc):
		desc = cap.pPackLenDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iPresetLut):
		desc = cap.pPresetLutDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iAeAlmSwDesc):
		desc = cap.pAeAlmSwDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iAeAlmHdDesc):
		desc = cap.pAeAlmHdDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iBayerDecAlmSwDesc):
		desc = cap.pBayerDecAlmSwDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iBayerDecAlmHdDesc):
		desc = cap.pBayerDecAlmHdDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )

def save_video_camera_init(
                      out_path,
                      fps:int=50,
                      name:str='video.avi',
                      video_size:tuple=(1280,1024),
                      codec:str='I420'
                      ):
    '''
    only for init, return out to write\n
    DONT FORGET RELEASE!!!\n
    Params
    @video_size:0 for camera_size,1 for manual size
    Notice:cv2.VideoCapture(1) can't work for video_save, I don't know why
    '''
    
    fourcc=cv2.VideoWriter_fourcc(*codec)
    final_path=os.path.join(out_path,name)
    out=cv2.VideoWriter(final_path,fourcc,fps,video_size)
    return out  

def for_trackbar(x):
    pass

def visualize_isp_config():
    '''
    only set windows and trackbars\n
    dont forget to call params in loop
    '''
    cv2.namedWindow('isp_config',cv2.WINDOW_NORMAL)
    cv2.createTrackbar("gamma", "isp_config", 0, 250,for_trackbar)
    cv2.createTrackbar("exposure_time_us", "isp_config", 10, 3000,for_trackbar)
    cv2.createTrackbar("r_gain", "isp_config", 0, 400,for_trackbar)
    cv2.createTrackbar("g_gain", "isp_config", 0, 400,for_trackbar)
    cv2.createTrackbar("b_gain", "isp_config", 0, 400,for_trackbar)
    cv2.createTrackbar("analog_gain", "isp_config", 64, 192,for_trackbar)
    cv2.createTrackbar("saturation", "isp_config", 0, 200,for_trackbar)
    cv2.createTrackbar("sharpness", "isp_config", 0, 100,for_trackbar)
    cv2.createTrackbar("contrast", "isp_config", 0, 200,for_trackbar)
    cv2.createTrackbar('set_default','isp_config',0,10,for_trackbar)
    cv2.createTrackbar('print_params','isp_config',0,10,for_trackbar)
    cv2.createTrackbar('load_params','isp_config',0,10,for_trackbar)

def visualize_filter_config():
    '''
    create trackbars for filter settings\n
    auto set default
    '''
    cv2.namedWindow('filter_config',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('area_range_min','filter_config',1,5000,for_trackbar)
    cv2.createTrackbar('area_range_max','filter_config',1,5000,for_trackbar)
    cv2.createTrackbar('normal_ratio','filter_config',101,300,for_trackbar)
    cv2.createTrackbar('strange_ratio1','filter_config',300,2000,for_trackbar)
    cv2.createTrackbar('strange_ratio2','filter_config',300,2000,for_trackbar)
    cv2.createTrackbar('shape_like_range_min','filter_config',20,180,for_trackbar)
    cv2.createTrackbar('shape_like_range_max','filter_config',20,180,for_trackbar)
    cv2.createTrackbar('center_dis_range_min','filter_config',1,1000,for_trackbar)
    cv2.createTrackbar('center_dis_range_max','filter_config',1,1000,for_trackbar)
    cv2.createTrackbar('print_params','filter_config',0,10,for_trackbar)
    cv2.createTrackbar('load_params','filter_config',0,10,for_trackbar)
    
    
        
    
    cv2.setTrackbarPos('area_range_min','filter_config',1)
    cv2.setTrackbarPos('area_range_max','filter_config',1000)
    cv2.setTrackbarPos('normal_ratio','filter_config',150)
    cv2.setTrackbarPos('strange_ratio1','filter_config',1000)
    cv2.setTrackbarPos('strange_ratio2','filter_config',500)
    cv2.setTrackbarPos('shape_like_range_min','filter_config',20)
    cv2.setTrackbarPos('shape_like_range_max','filter_config',180)
    cv2.setTrackbarPos('center_dis_range_min','filter_config',200)
    cv2.setTrackbarPos('center_dis_range_max','filter_config',300)
    cv2.setTrackbarPos('print_params','filter_config',0)
    cv2.setTrackbarPos('load_params','filter_config',0)
    
    
    


def track_bar_set_default():
    '''
    after visualize isp_config\n
    use this to set trackbar position
    '''
    cv2.setTrackbarPos('gamma','isp_config',30)
    cv2.setTrackbarPos('exposure_time_us','isp_config',100) 
    cv2.setTrackbarPos('r_gain','isp_config',100) 
    cv2.setTrackbarPos('g_gain','isp_config',100) 
    cv2.setTrackbarPos('b_gain','isp_config',100) 
    cv2.setTrackbarPos('analog_gain','isp_config',64) 
    cv2.setTrackbarPos('saturation','isp_config',100) 
    cv2.setTrackbarPos('contrast','isp_config',100) 
    cv2.setTrackbarPos('sharpness','isp_config',0)
    cv2.setTrackbarPos('set_default','isp_config',0)
    cv2.setTrackbarPos('print_params','isp_config',0)
    cv2.setTrackbarPos('load_params','isp_config',0)

    
  
def print_params():
    '''
    only use in isp_config window
    '''
    
    gamma=cv2.getTrackbarPos('gamma','isp_config')
    exposure_time_us=cv2.getTrackbarPos('exposure_time_us','isp_config')
    r_gain=cv2.getTrackbarPos('r_gain','isp_config')
    g_gain=cv2.getTrackbarPos('g_gain','isp_config')
    b_gain=cv2.getTrackbarPos('b_gain','isp_config')
    analog_gain=cv2.getTrackbarPos('analog_gain','isp_config')
    saturation=cv2.getTrackbarPos('saturation','isp_config')
    sharpness=cv2.getTrackbarPos('sharpness','isp_config')
    contrast=cv2.getTrackbarPos('contrast','isp_config')
    print('-------------ISP_params-----------')
    print('gamma:',gamma)
    print('exposure_time_us:',exposure_time_us)
    print('rgb_gain:',(r_gain,g_gain,b_gain))
    print('analog_gain:',analog_gain)
    print('saturation:',saturation)
    print('sharpness:',sharpness)
    print('contrast:',contrast)
    print('-------------End---------------')

def trackbar_set_group(group):
    '''
    Important:\n
    @Params order:
    @gamma
    @exposure_time_us
    @r_gain
    @g_gain
    @b_gain
    @analog_gain
    @saturation
    @sharpness
    @contrast
    
    
    '''
    global isp_params_list
    if len(isp_params_list)==0:
        return 0
    group=group-1
    if 0<=group<len(isp_params_list):
        cv2.setTrackbarPos('gamma','isp_config',isp_params_list[group][0])
        cv2.setTrackbarPos('exposure_time_us','isp_config',isp_params_list[group][1]) 
        cv2.setTrackbarPos('r_gain','isp_config',isp_params_list[group][2]) 
        cv2.setTrackbarPos('g_gain','isp_config',isp_params_list[group][3]) 
        cv2.setTrackbarPos('b_gain','isp_config',isp_params_list[group][4]) 
        cv2.setTrackbarPos('analog_gain','isp_config',isp_params_list[group][5]) 
        cv2.setTrackbarPos('saturation','isp_config',isp_params_list[group][6]) 
        cv2.setTrackbarPos('sharpness','isp_config',isp_params_list[group][7]) 
        cv2.setTrackbarPos('contrast','isp_config',isp_params_list[group][8])
    
    

    
def trackbar_set_isp(hcamera):
    set_default=cv2.getTrackbarPos('set_default','isp_config')
    params_print=cv2.getTrackbarPos('print_params','isp_config')
    load_params=cv2.getTrackbarPos('load_params','isp_config')
    
    
    if  load_params!=0:
        trackbar_set_group(load_params)
    if set_default==5:
        track_bar_set_default()
    if params_print==5:
        print_params()
        cv2.setTrackbarPos('print_params','isp_config',0)
    
    gamma=cv2.getTrackbarPos('gamma','isp_config')
    exposure_time_us=cv2.getTrackbarPos('exposure_time_us','isp_config')
    r_gain=cv2.getTrackbarPos('r_gain','isp_config')
    g_gain=cv2.getTrackbarPos('g_gain','isp_config')
    b_gain=cv2.getTrackbarPos('b_gain','isp_config')
    analog_gain=cv2.getTrackbarPos('analog_gain','isp_config')
    saturation=cv2.getTrackbarPos('saturation','isp_config')
    sharpness=cv2.getTrackbarPos('sharpness','isp_config')
    contrast=cv2.getTrackbarPos('contrast','isp_config')
    
    
    try:
        set_isp(hcamera,exposure_time_us,gamma,r_gain,g_gain,b_gain,saturation=saturation,contrast=contrast,sharpness=sharpness,analog_gain=analog_gain)
    except:
        print('set_isp failed')

def trackbar_set_filter()->tuple:
    global filter_params_list
    params_print=cv2.getTrackbarPos('print_params','filter_config')
    params_load=cv2.getTrackbarPos('load_params','filter_config')
    if params_load !=0 and len(filter_params_list)>=params_load:
        params_load-=1
        cv2.setTrackbarPos('area_range_min','filter_config',filter_params_list[params_load][0][0])
        cv2.setTrackbarPos('area_range_max','filter_config',filter_params_list[params_load][0][1])
        cv2.setTrackbarPos('normal_ratio','filter_config',filter_params_list[params_load][1])
        cv2.setTrackbarPos('strange_ratio1','filter_config',filter_params_list[params_load][2])
        cv2.setTrackbarPos('strange_ratio2','filter_config',filter_params_list[params_load][3])
        cv2.setTrackbarPos('shape_like_range_min','filter_config',filter_params_list[params_load][4][0])
        cv2.setTrackbarPos('shape_like_range_max','filter_config',filter_params_list[params_load][4][1])
        cv2.setTrackbarPos('center_dis_range_min','filter_config',filter_params_list[params_load][5][0])
        cv2.setTrackbarPos('center_dis_range_max','filter_config',filter_params_list[params_load][5][1])
        cv2.setTrackbarPos('print_params','filter_config',0)      
    area_range=(cv2.getTrackbarPos('area_range_min','filter_config'),cv2.getTrackbarPos('area_range_max','filter_config'))
    normal_ratio=cv2.getTrackbarPos('normal_ratio','filter_config')/100
    strange_ratio1=cv2.getTrackbarPos('strange_ratio1','filter_config')/100
    strange_ratio2=cv2.getTrackbarPos('strange_ratio2','filter_config')/100
    shape_like_range=(cv2.getTrackbarPos('shape_like_range_min','filter_config')/100,cv2.getTrackbarPos('shape_like_range_max','filter_config')/100)
    center_dis_range=(cv2.getTrackbarPos('center_dis_range_min','filter_config'),cv2.getTrackbarPos('center_dis_range_max','filter_config'))
    if params_print==5:
        cv2.setTrackbarPos('print_params','filter_config',0)
        print('^^^^^^^^^^^^^filter_params^^^^^^^^^^^^^')
        print('area_range:',area_range)
        print('normal_ratio:',normal_ratio)
        print('strange_ratio1:',strange_ratio1)
        print('strange_ratio2:',strange_ratio2)
        print('shape_like_range:',shape_like_range)
        print('center_dis_range:',center_dis_range)
        print('^^^^^^^^^^^^^^^^^^^end^^^^^^^^^^^^^^^^^^^^^^^')
    
    return area_range,normal_ratio,strange_ratio1,strange_ratio2,shape_like_range,center_dis_range
    

