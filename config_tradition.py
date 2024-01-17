from autoaim_alpha.autoaim_alpha.camera.mv_class import *
from autoaim_alpha.autoaim_alpha.img import detector
from autoaim_alpha.autoaim_alpha.img.tools import add_text
armor_color = 'red'
tradition_config_path = './tradition_config'
mode = 'Dbg'

if __name__ == "__main__":
    
    ca = Mindvision_Camera(output_format='BGR8',
                           if_auto_exposure=False,
                           if_trigger_by_software=False,
                           if_use_last_params=False,
                           pingpong_exposure=None,
                           camera_mode=mode,
                           tradition_config_folder_path=tradition_config_path,
                           armor_color=armor_color)
    
    fps = 0
    
    
    ca.print_show_params()
    ca.enable_trackbar_config(press_key_to_save='a')
    
    
    tradition_detector = detector.Tradition_Detector(armor_color=armor_color,
                                                     mode=mode,
                                                     roi_single_shape=[32,32],
                                                     tradition_config_folder_path=tradition_config_path
                                                       
                                                       )
    
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
            
            add_text(img_ori,'FPS',fps,scale_size=1)
            cv2.imshow('ori',img_ori)
            print(fps)
            
            ca.detect_trackbar_config()
            tradition_detector.detect_trackbar_config()
            tradition_detector.filter1.detect_trackbar_config()
            tradition_detector.filter2.detect_trackbar_config()
            

            t2 = time.perf_counter()
            fps =round( 1/(t2 - t1))
        
        
    cv2.destroyAllWindows()
        
        