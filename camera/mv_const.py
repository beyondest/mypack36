import mvsdk


CAMERA_TYPE_TO_SHOW_DICT =    {mvsdk.CAMERA_MEDIA_TYPE_RGB8:"RGB8",
                               mvsdk.CAMERA_MEDIA_TYPE_BGR8:"BGR8",
                               mvsdk.CAMERA_MEDIA_TYPE_YUV8_UYV:"YUV8",
                               mvsdk.CAMERA_MEDIA_TYPE_BAYBG8:"BAYBG8",
                               mvsdk.CAMERA_MEDIA_TYPE_BAYGB8:"BAYGB8"
                              }
CAMERA_SHOW_TO_TYPE_DICT = {v:k for k,v in CAMERA_TYPE_TO_SHOW_DICT.items()}
CAMERA_TRACKBAR_TO_TYPE_DICT = {k:v for k,(v,_) in enumerate(CAMERA_TYPE_TO_SHOW_DICT.items())}
       
       
ISP_PARAMS_SCOPE_LIST     = [[10,300000],   # Exposure_u4s  0
                                [0,250],    # Gamma         1
                                [0,400],    # R_gain        2
                                [0,400],    # G_gain        3
                                [0,400],    # B_gain        4    
                                [64,192],   # analog_gain   5
                                [0,100],    # sharpness     6
                                [0,200],    # saturation    7
                                [0,200],    # contrast      8
                                [10,1280],  # wid           9
                                [10,1024]   # hei           10
                                ]

CAMERA_RESOLUTION_DEFAULT_XY = (1280,1024)

