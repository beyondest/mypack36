from camera import mvsdk


CAMERA_TYPE_TO_SHOW_DICT =    {mvsdk.CAMERA_MEDIA_TYPE_RGB8:"RGB8",
                               mvsdk.CAMERA_MEDIA_TYPE_BGR8:"BGR8",
                               mvsdk.CAMERA_MEDIA_TYPE_YUV8_UYV:"YUV8",
                               mvsdk.CAMERA_MEDIA_TYPE_BAYBG8:"BAYBG8",
                               mvsdk.CAMERA_MEDIA_TYPE_BAYGB8:"BAYGB8"
                              }
CAMERA_SHOW_TO_TYPE_DICT = {v:k for k,v in CAMERA_TYPE_TO_SHOW_DICT.items()}
CAMERA_TRACKBAR_TO_TYPE_DICT = {k:v for k,(v,_) in enumerate(CAMERA_TYPE_TO_SHOW_DICT.items())}
       
       
ISP_PARAMS_SCOPE_LIST     = [[10,30000],   # Exposure_u4s  0
                                [0,250],    # Gamma         1
                                [0,400],    # R_gain        2
                                [0,400],    # G_gain        3
                                [0,400],    # B_gain        4    
                                [64,192],   # analog_gain   5
                                [0,100],    # sharpness     6
                                [0,200],    # saturation    7
                                [0,200],    # contrast      8
                                [10,1280],  # resolution_x  9,
                                [10,1024],  # resolution_y  10
                                ]

CAMERA_CHANNEL_NUMS = 3
CAMERA_ALIGN_BYTES_NUMS = 16
CAMERA_GRAB_IMG_WATI_TIME_MS = 1000

# Default Isp Params
CAMERA_EXPOSURE_TIME_US_DEFALUT = 10*1000
CAMERA_GAMMA_DEFAULT = 30
CAMERA_GAIN_DEFAULT = [100,100,100]
CAMERA_ANALOG_GAIN_DEFAULT =64
CAMERA_SHARPNESS_DEFAULT = 0
CAMERA_SATURATION_DEFAULT = 100
CAMERA_CONTRAST_DEFAULT = 100
CAMERA_ROI_RESOLUTION_DEFAULT_XY = [1280,1024]
CAMERA_RESOLUTION_DEFAULT_XY = [1280,1024]





