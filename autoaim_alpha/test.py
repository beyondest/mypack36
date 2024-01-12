
from autoaim_alpha.camera.mv_class import *
import autoaim_alpha.img.img_operation as imo

if __name__ == "__main__":
    
    ca = Mindvision_Camera(if_trigger_by_software=False,if_use_default_params=True)
    fps = 0
    #ca.enable_trackbar_config('config')
    with Custome_Context('camera',ca):
        while 1:
            
            t1 = time.perf_counter()
            

            img = ca.get_img_continous()
            
            t2 = time.perf_counter()
            
            t = t2 - t1
            fps = round(1/t)
            
            #ca.detect_trackbar_actions_when_isp_config()
            imo.add_text(img,'FPS',fps,scale_size=1)
            
            cv2.imshow('h',img)
            
            
            
          
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        
        
    cv2.destroyAllWindows()
        
        