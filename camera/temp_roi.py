#coding=utf-8
import cv2
import sys
import img.img_operation as imo
import os
import control
import sys
out_path='D:/tmdvs/pywork/vsRM/test'
armor_color='red'

def for_trackbar(x):
    pass
 


hcamera=control.camera_init()
control.isp_init(hcamera)
#out=control.save_video_camera_init(out_path,name='out.mp4',codec='AVC1')


camera_info=control.get_all(hcamera)
control.print_getall(camera_info)


control.camera_open(hcamera)


cv2.namedWindow('w1',cv2.WINDOW_NORMAL)
cv2.namedWindow('w2',cv2.WINDOW_NORMAL)
cv2.namedWindow('w3',cv2.WINDOW_NORMAL)



control.visualize_isp_config()
control.track_bar_set_default()
control.visualize_filter_config()

pframebuffer_address=control.camera_setframebuffer()
#press esc to end
while (cv2.waitKey(1) & 0xFF) != 27:
    control.trackbar_set_isp(hcamera)
    filter_params=control.trackbar_set_filter()
    
    time1=cv2.getTickCount()
    
    img_ori=control.grab_img(hcamera,pframebuffer_address)
    
    

    img_ori=cv2.resize(img_ori,(320,256),interpolation=cv2.INTER_AREA)
    
    
    
    
    
    
    
    
    img_single,t1=imo.pre_process(img_ori,armor_color)
    
    
    big_rec_list,t2=imo.find_big_rec_plus(img_single,filter_params)
    img_bgr=imo.draw_big_rec(big_rec_list,img_ori)
    
    roi_list,t3=imo.pick_up_roi(big_rec_list,img_ori)
    roi_single_list,t4=imo.pre_process2(roi_list,armor_color)
    
    t1=round(t1,6)
    t2=round(t2,6)
    t3=round(t3,6)
    t4=round(t4,6)
    time2=cv2.getTickCount()
    time_all=(time2-time1)/cv2.getTickFrequency()
    fps=round(1/time_all)
    if img_ori.shape==(1024,1280,3):
        img_bgr=imo.add_text(img_bgr,'preprecess1_time',t1,(100,50))
        img_bgr=imo.add_text(img_bgr,'preprocess2_time',t4,(100,100))
        img_bgr=imo.add_text(img_bgr,'find_bigrec_time',t2,(100,150))
        img_bgr=imo.add_text(img_bgr,'pick_roi_time',t3,(100,200))
        img_bgr=imo.add_text(img_bgr,'fps',fps,(100,250))
        
    elif img_ori.shape==(256,320,3):
        img_bgr=imo.add_text(img_bgr,'preprecess1_time',t1,(25,12))
        img_bgr=imo.add_text(img_bgr,'preprocess2_time',t4,(25,25))
        img_bgr=imo.add_text(img_bgr,'find_bigrec_time',t2,(25,37))
        img_bgr=imo.add_text(img_bgr,'pick_roi_time',t3,(25,49))
        img_bgr=imo.add_text(img_bgr,'fps',fps,(25,62))
        
    else:
        print('size wrong, look at test.py')
        sys.exit()
    #out.write(dst)
    
    
    cv2.imshow('w1',img_bgr) 
    cv2.imshow('w2',img_single)
    if len(roi_single_list)==1:
        cv2.imshow('w3',roi_single_list[0])
cv2.destroyAllWindows()
control.camera_close(hcamera,pframebuffer_address)
#out.release()

