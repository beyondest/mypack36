from autoaim_alpha.camera.mv_class import *
from autoaim_alpha.img import img_operation as imo


if __name__ == "__main__":
    
    ca = Mindvision_Camera(output_format='BGR8',
                           if_auto_exposure=False,
                           if_trigger_by_software=False,
                           if_use_default_params=False,
                           pingpong_exposure=None,
                           camera_mode='Dbg')
    
    fps = 0
    ca.save_custom_params_to_yaml('./isp.yaml')
    ca.save_all_params_to_file('./all')
    ca.print_show_params()

    with Custome_Context('camera',ca):
        
        while 1:
            
            t1 = time.perf_counter()
            
            img_ori= ca.get_img()
            
            imo.add_text(img_ori,'FPS',fps,scale_size=1)
            cv2.imshow('ori',img_ori)
            print(fps)
            #ca.detect_trackbar_actions_when_isp_config()
                
                
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            
            t2 = time.perf_counter()
            fps =round( 1/(t2 - t1))
        
        
    cv2.destroyAllWindows()
        
        