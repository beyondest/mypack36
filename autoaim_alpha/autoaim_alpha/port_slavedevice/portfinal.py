import serial
import time
import threading

import sys
sys.path.append('..')
sys.path.append('../camera')
from os_op.thread_op import *
from com_tools import *
import img.img_operation as imo
import cv2
from motor_params import *
import camera.control as cac
import camera.mvsdk as mvsdk
from utils_network.mymath import map_value


class Param:
    def __init__(self) -> None:
        
        self.img = np.zeros((1024,1280))
        self.adata = action_data()
        self.pdata = pos_data()
        self.sdata = syn_data()
        self.draw_circle_counts = 0
        

def read_and_show(ser:serial.Serial,
                  pdata:pos_data):
    read_ori = read_data(ser)
    if read_ori == b'':
        print("Nothing receive")
    else:
        #print(f'receiving: {read_ori}')
        
        if_error =pdata.convert_pos_bytes_to_data(read_ori,if_part_crc=False)
        #print(f' if_error:{if_error}, crc_get:{pdata.crc_get},my_crc:{pdata.crc_v}')
        #pdata.show()
    
    
def just_read(ser:serial.Serial):
    read_ori = read_data(ser)
    print(f'receiving: {read_ori}')
    
    
    
def write_and_show(ser:serial.Serial,
                   adata:action_data,
                   sdata:syn_data,
                   s_or_a:str ,
                   windows_name:str,
                   debug_trackbar:str,
                   rel_pitch_trackbar:str,
                   rel_yaw_trackbar:str
                   ):
    
   # adata.setting_voltage_or_rpm = round(cv2.getTrackbarPos(debug_trackbar,windows_name) - debug_track_bar_scope[1]/2)
   # adata.target_pitch_10000 =round(cv2.getTrackbarPos(rel_pitch_trackbar,windows_name) - pitch_radians_scope[1]/2)
   # adata.target_yaw_10000 = round(cv2.getTrackbarPos(rel_yaw_trackbar,windows_name) - yaw_radians_scope[1]/2)
    
    a_towrite = adata.convert_action_data_to_bytes(if_part_crc=False)
    
    s_towrite = sdata.convert_syn_data_to_bytes(if_part_crc=False)
    
    if s_or_a == 's':
        
        #print(f"Writing: {s_towrite}")
        #print(f"crcis:{sdata.crc_v}")
        ser.write(s_towrite)
    elif s_or_a == 'a':
        
        #print(f'Writing:{a_towrite}')
        #print(f"crcis:{adata.crc_v}")
        
        ser.write(a_towrite)
    
def show_everything(param:Param,
                    windows_name:str,
                    ):
    
    txt_x_step = 175
    txt_y_step = 20
    param.img = cv2.flip(param.img,0)
    
    txt_x =1
    for i in range(param.adata.len):
        param.img = imo.add_text(param.img,param.adata.label_list[i],param.adata.list[i],(txt_x,txt_y_step),scale_size=0.7)
        txt_x += txt_x_step
        
    txt_x =1    
    
    for i in range(param.pdata.len):
        param.img = imo.add_text(param.img,param.pdata.label_list[i],param.pdata.list[i],(txt_x,txt_y_step*2),scale_size=0.7)
        txt_x+=txt_x_step

    
    txt_x = 1
    for i in range(param.sdata.len):
        param.img = imo.add_text(param.img,param.sdata.label_list[i],param.sdata.list[i],(txt_x,txt_y_step*3),scale_size=0.7)
        txt_x+=txt_x_step
    
    param.img = imo.draw_crosshair(param.img)
    cv2.imshow(windows_name,param.img)
    
    
def show_deinit():
    cv2.destroyAllWindows()
    

def convert_xy_to_relative_radians_10000(x,y,shape_of_img:tuple):
    """
    Returns:
        target_pitch_10000,target_yaw_10000
    """
    
    target_pitch_10000 = round(map_value(y,(0,shape_of_img[0]),(-15608,15708)))
    target_yaw_10000 = round(map_value(x,(0,shape_of_img[1]),(-31416,31416)))
    return target_pitch_10000,target_yaw_10000
    

    
def cv2_click_event_callback(event,x,y,flags,param:Param):
    if event == cv2.EVENT_LBUTTONUP:
        
        print('fuck')
        param.adata.target_pitch_10000 = round(map_value(y,(0,1024),(-15708,15708)))
        param.adata.target_yaw_10000 = round(map_value(x,(0,1280),(-31416,31416)))
        param.draw_circle_counts = 0
        
    
      


      

       
        



if __name__ == "__main__":
    
    ser = port_open(port_abs_path='COM3')
    hcamera =cac.camera_init(out_put_format=mvsdk.CAMERA_MEDIA_TYPE_BGR8)
    cac.set_isp(hcamera)
    camera_info = cac.get_all(hcamera)
    cac.print_getall(camera_info)
    pframe_buffer_addr = cac.camera_setframebuffer()
    cac.camera_open(hcamera)
    
    debug_track_bar = 'tar_rpm'
    rel_pitch_trackbar = 'rel_rad_pit'
    rel_yaw_trackbar = 'rel_rad_yaw'
    windows_name = 'camera_show'
    
    AUTO_AIM = True
    
    time_show_x_count = 0
    global_param = Param()
    global_param.img = cac.grab_img(hcamera,pframe_buffer_addr)
    cv2.namedWindow(windows_name,cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(windows_name,cv2_click_event_callback,global_param)
    imo.trackbar_init(debug_track_bar,debug_track_bar_scope,windows_name)
    imo.trackbar_init(rel_pitch_trackbar,pitch_radians_scope,windows_name)
    imo.trackbar_init(rel_yaw_trackbar,yaw_radians_scope,windows_name)
    
    cv2.setTrackbarPos(debug_track_bar,windows_name,int(debug_value_start_pos+debug_track_bar_scope[1]/2))
    cv2.setTrackbarPos(rel_pitch_trackbar,windows_name,int(pitch_start_pos+pitch_radians_scope[1]/2))
    cv2.setTrackbarPos(rel_yaw_trackbar,windows_name,int(yaw_start_pos+yaw_radians_scope[1]/2))

    task1 = task(0.05,
                 read_and_show,
                 port_close,
                 [ser,global_param.pdata],
                 [ser])
    
    
    task2 = task(0.05,
                 for_circle_func=write_and_show,
                 params_for_circle_func=[ser,global_param.adata,global_param.sdata,'a',windows_name,debug_track_bar,rel_pitch_trackbar,rel_yaw_trackbar],
                 )
    
    task1.start()
    task2.start()
    
    while 1:
        global_param.img = cac.grab_img(hcamera,pframe_buffer_addr)
        
        center_list = imo.find_armor_beta(global_param.img)
        if AUTO_AIM:
            if len(center_list) == 1:
                global_param.adata.target_pitch_10000 , global_param.adata.target_yaw_10000 = \
                    convert_xy_to_relative_radians_10000(center_list[0][0],center_list[0][1],(1024,1280))
        
        show_everything(global_param,windows_name)
        
        if cv2.waitKey(1) == ord('q'):
            
            show_deinit()
            task1.end()
            task2.end()
            cac.camera_close(hcamera,pframe_buffer_addr)
            break
    
    
    
    
    
    
    
    
    


