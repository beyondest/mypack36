
from autoaim_alpha.autoaim_alpha.camera.mv_class import Mindvision_Camera
from autoaim_alpha.autoaim_alpha.img.detector import Armor_Detector
from autoaim_alpha.autoaim_alpha.os_op.basic import *
from autoaim_alpha.autoaim_alpha.img.tools import *

import time

armor_color = 'blue'
mode = 'Dbg'
if_yolvo5 = False
camera_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/camera_config' 
tradition_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/tradition_config'
net_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/net_config'
depth_estimator_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/other_config/pnp_params.yaml'


fps = 0

if __name__ == '__main__':
    
    ca = Mindvision_Camera(        
                            output_format='bgr8',
                            camera_mode=mode,
                            camera_config_folder=camera_config_folder,
                            armor_color=armor_color,
                            if_yolov5=if_yolvo5
                            )
    
    ca.print_show_params()
    
    de = Armor_Detector(
                        armor_color=armor_color,
                        mode=mode,
                        tradition_config_folder=tradition_config_folder,
                        net_config_folder=net_config_folder,
                        save_roi_key='c',
                        depth_estimator_config_yaml=depth_estimator_config_yaml_path,
                        if_yolov5=if_yolvo5
                        )
    
    #ca.enable_trackbar_config(press_key_to_save='a')
    #de.tradition_detector.enable_preprocess_config(press_key_to_save='s')
    #de.tradition_detector.filter1.enable_trackbar_config(press_key_to_save='d')
    #de.tradition_detector.filter2.enable_trackbar_config(press_key_to_save='f')
    
    #de.tradition_detector.enable_save_roi('/mnt/d/datasets/autoaim/roi_binary/based',save_interval=3)
    #ca.enable_save_img('/mnt/d/datasets/autoaim/camera_img/red/based',save_img_interval=3)
    #ca.enbable_save_video('/mnt/d/datasets/autoaim/video2.mp4',fps=30)
    
    #de.depth_estimator.enable_trackbar_config(save_params_key='n')
    
    with Custome_Context('camera',ca):
        
        while True:
            
            t1 = time.perf_counter()
            
            img = ca.get_img()
            
            t2 = time.perf_counter()
            print(f'get_img time:{t2-t1}')
            
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            result_list,t = de.get_result(img)

            t3 = time.perf_counter()
            print(f'get_result_time:{t3-t2}')
            
            
            de.visualize(img,fps,windows_name='result')
                    
            
            #ca._detect_trackbar_config()
            #de.tradition_detector._detect_trackbar_config()
            #de.tradition_detector.filter1._detect_trackbar_config()
            #de.tradition_detector.filter2._detect_trackbar_config()
            
            t4 = time.perf_counter()
            fps = round(1/(t4-t1))
            
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
    
    cv2.destroyAllWindows()
            
            
            
        
        
        
        
    