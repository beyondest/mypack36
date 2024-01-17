from autoaim_alpha.autoaim_alpha.camera.mv_class import Mindvision_Camera
from autoaim_alpha.autoaim_alpha.img.detector import Armor_Detector
from autoaim_alpha.autoaim_alpha.os_op.basic import *
from autoaim_alpha.autoaim_alpha.img.tools import *

import time

armor_color = 'red'
mode = 'Dbg'
tra_config_folder = './tmp_tradition_config'
net_config_folder = './tmp_net_config'
custom_isp_params_yaml = './tradition_config/red/custom_isp_params.yaml'


fps = 0

if __name__ == '__main__':
    
    ca = Mindvision_Camera(        
                                   output_format='bgr8',
                                   camera_mode=mode,
                                   custom_isp_yaml_path=custom_isp_params_yaml,
                                   armor_color=armor_color
                                   )
    
    ca.print_show_params()
    
    de = Armor_Detector(
                        armor_color=armor_color,
                        mode=mode,
                        tradition_config_folder=tra_config_folder,
                        net_config_folder=net_config_folder,
                        save_roi_key='c'
                        )
    
    ca.enable_trackbar_config(press_key_to_save='a')
    de.tradition_detector.enable_preprocess_config(press_key_to_save='s')
    de.tradition_detector.filter1.enable_trackbar_config(press_key_to_save='d')
    de.tradition_detector.filter2.enable_trackbar_config(press_key_to_save='f')
    
    
    
    
    with Custome_Context('camera',ca):
        
        while True:
            
            t1 = time.perf_counter()
            
            img = ca.get_img()
            
            t2 = time.perf_counter()
            print(f'get_img time:{t2-t1}')
            
            result_list,t = de.get_result(img,img)
            
            t3 = time.perf_counter()
            print(f'get_result_time:{t3-t2}')
            
            de.visualize(img,fps,windows_name='result')

                    
            ca.detect_trackbar_config()
            de.tradition_detector.detect_trackbar_config()
            de.tradition_detector.filter1.detect_trackbar_config()
            de.tradition_detector.filter2.detect_trackbar_config()
            
            
            
            
            
            t4 = time.perf_counter()
            fps = round(1/(t4-t1))
            
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
    
    cv2.destroyAllWindows()
            
        
        
        
        
    