from autoaim_alpha.camera.mv_class import *
from autoaim_alpha.img import img_operation as imo


if __name__ == "__main__":
    
    ca = Mindvision_Camera(if_auto_exposure=False,
                           if_trigger_by_software=False,
                           if_use_default_params=True,
                           pingpong_exposure=[1*1000,10*1000])
    fps = 0
    t11 = 0
    t21 = 0
    #ca.enable_trackbar_config('config')
    with Custome_Context('camera',ca):
        while 1:
            
            

            img,count = ca.get_img_pingpong_exposure()
            
            #ca.detect_trackbar_actions_when_isp_config()
           
            
            if count % 2:
                
                t12 = time.perf_counter()
                fps = round(1/(t12-t11))
                imo.add_text(img,'FPS',fps,scale_size=1)
                
                cv2.imshow('exposure_1',img)
                t11 = t12
            
            
            else:
                t22 = time.perf_counter()
                fps = round(1/(t22-t21))
                imo.add_text(img,'FPS',fps,scale_size=1)
                
                cv2.imshow('exposure_0',img)
                t21 = t22
                
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        
        
    cv2.destroyAllWindows()
        
        