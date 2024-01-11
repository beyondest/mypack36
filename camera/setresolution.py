import sys
sys.path.append('..')
import camera.mvsdk as mvsdk
import camera.control as cac
h = cac.camera_init()
out =mvsdk.CameraGetFriendlyName(h)

res = mvsdk.CameraGetImageResolution(h)
print(res)
mvsdk.CAMERA_MEDIA_TYPE_RGB8    # 35127316      Default
mvsdk.CAMERA_MEDIA_TYPE_BAYBG8  # 17301515
mvsdk.CAMERA_MEDIA_TYPE_YUV8_UYV  # 35127328
mvsdk.CAMERA_MEDIA_TYPE_BAYGB8  # 17301514
mvsdk.CAMERA_MEDIA_TYPE_BGR8    # 35127317
mvsdk.tSdkImageResolution       #iWidthFOV, iHeightFOV;iWidthZoomSw:1000,iHeightZoomSw:100

#320 256 640 512

if 1:
    r = mvsdk.CameraSetImageResolutionEx(       hCamera=h,
                                            iIndex=0xff,
                                            Mode=0,
                                            ModeSize=0,
                                            x=320,
                                            y=256,
                                            width=640,
                                            height=512,
                                            ZoomWidth=1000,
                                            ZoomHeight=100)
    print(r)

res = mvsdk.CameraGetImageResolution(h)
print(res)