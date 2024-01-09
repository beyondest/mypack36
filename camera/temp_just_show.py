import control
import cv2
import mvsdk
import numpy as np
import time
import sys
sys.path.append('..')
    
import img.img_operation as imo
class Param:
    def __init__(self) -> None:
        self.img = np.zeros((100,100))

def mouse_callback(event,x,y,flags,param:Param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param.img,(x,y),10,(255,0,0),-1)

pp = Param()

cv2.namedWindow('camera',cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('camera',mouse_callback,pp)

hcamera = control.camera_init(out_put_format=mvsdk.CAMERA_MEDIA_TYPE_BGR8)
control.set_isp(hcamera)
camera_info=control.get_all(hcamera)
control.print_getall(camera_info)
#out=control.save_video_camera_init(out_path,name='out.mp4',codec='AVC1')
control.camera_open(hcamera)
pframebuffer_address=control.camera_setframebuffer()


while (cv2.waitKey(1) & 0xFF) != 27:
    
    t1 = time.perf_counter()
    pp.img=control.grab_img(hcamera,pframebuffer_address)
    pp.img = cv2.flip(pp.img,0)
    #out.write(dst)
    pp.img = imo.draw_crosshair(pp.img)
    cv2.imshow('camera',pp.img) 
    t2 = time.perf_counter()
    t = t2-t1
    fps = round(1/t)
    print('FPS:',fps)
    
cv2.destroyAllWindows()

control.camera_close(hcamera,pframebuffer_address)

#out.release()


