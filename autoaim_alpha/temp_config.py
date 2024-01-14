from autoaim_alpha.camera.mv_class import *
from autoaim_alpha.img import img_operation as imo
from autoaim_alpha.img import detector

if __name__ == "__main__":
    
    ca = Mindvision_Camera(output_format='BGR8',
                           if_auto_exposure=False,
                           if_trigger_by_software=False,
                           if_use_default_params=False,
                           pingpong_exposure=None,
                           camera_mode='Dbg')
    
    fps = 0
    #ca.save_custom_params_to_yaml('./isp.yaml')
    #ca.save_all_params_to_file('./all')
    ca.load_params_from_yaml('./tradition_config/blue/custom_isp_params.yaml')
    ca.print_show_params()
    ca.enable_trackbar_config(press_key_to_save='a')
    
    
    tradition_detector = detector.Traditional_Detector('blue','Dbg',True)
    
    tradition_detector.enable_preprocess_config(press_key_to_save='s')
    tradition_detector.filter1.enable_trackbar_config(press_key_to_save='d')
    tradition_detector.filter2.enable_trackbar_config(press_key_to_save='f')
    
    with Custome_Context('camera',ca):
        
        while 1:
            
            t1 = time.perf_counter()
            
            img_ori= ca.get_img()
            
            tradition_detector.get_output(
                                            img_ori,
                                            img_ori
                                          )
            
            imo.add_text(img_ori,'FPS',fps,scale_size=1)
            cv2.imshow('ori',img_ori)
            print(fps)
            
            ca.detect_trackbar_actions_when_isp_config()
            tradition_detector.detect_trackbar_config()
            tradition_detector.filter1.detect_trackbar_config()
            tradition_detector.filter2.detect_trackbar_config()
            

            t2 = time.perf_counter()
            fps =round( 1/(t2 - t1))
        
        
    cv2.destroyAllWindows()
        
        