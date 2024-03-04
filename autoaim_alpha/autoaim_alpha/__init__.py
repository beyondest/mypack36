import rclpy.qos
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from . import os_op
from .os_op.basic import *
from .utils_network.data import Data
from pyquaternion import Quaternion
from visualization_msgs.msg import Marker

#from autoaim_interface.msg import *
from .haha import *



#from .haha import *
armor_type_list = ['1d','2x','3d','3x','4d','4x','5d','5x','based','basex','sentry']

general_config = Data.get_file_info_from_yaml('/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/general_config.yaml')
mode = general_config['mode']
armor_color = general_config['armor_color']

enemy_car_list=  [{'armor_name':'3x','armor_distance':[0.4,0.5],'armor_nums': 4},
                               {'armor_name':'4x','armor_distance': [0.5,0.5],'armor_nums': 2}]


                                                    # Public Params
car_rotation_axis = np.array([0,1,0])
# detect_delay + correct_and_predict_delay + cal_ballistic_delay + com_delay
total_delay = 0.03 + 0.01 + 0.01 + 0.005
predict_time_offset = 0.3

topic_img_raw = {'name': 'img_raw', 'type': Image, 'qos_profile':3}
topic_img_detected = {'name': 'img_detected', 'type': Image, 'qos_profile':3}
topic_electric_sys_state = {'name': 'electric_sys_state', 'type':ElectricsysState , 'qos_profile':10}
topic_detect_result = {'name': 'detect_result', 'type': DetectResult, 'qos_profile':10}
topic_armor_pos_without_correct = {'name': 'armor_pos_without_correct', 'type': ArmorPos, 'qos_profile':10}
topic_armor_pos_corrected = {'name': 'armor_pos_corrected', 'type': ArmorPos, 'qos_profile':10}
topic_armor_pos_predicted = {'name': 'armor_pos_predicted', 'type': ArmorPos, 'qos_profile':10}
topic_car_pos = {'name': 'car_pos', 'type': CarPos, 'qos_profile':10}
topic_electric_sys_com = {'name': 'electric_sys_com', 'type': ElectricsysCom, 'qos_profile':10}

topic_marker_armor_pos_corrected = {'name':'marker', 'type': Marker, 'qos_profile':10}

if_pub_car_state = True
if_pub_armor_state_without_correct = True
if_pub_armor_state_corrected = True
if_pub_armor_state_predicted = False
if_pub_img_detected = True
                                                    # Node Webcam Mv
                                                    
node_webcam_mv_name = 'node_webcam_mv'
node_webcam_mv_frequency = 30


camera_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/camera_config' 
tradition_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/tmp_tradition_config'


camera_output_format = 'bgr8'





                                                
                                                    # Node Detector


node_detector_name = 'node_detect'

tradition_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/tmp_tradition_config'
net_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/tmp_net_config'




                                                    # Node Observer
                                                    
node_observer_name = 'node_observer'
observer_correct_freq = 100
observer_predict_freq = 100


observer_predict_freq = 100
observer_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/observer_config.yaml'


                                                  
                                                    # Node Decision Maker
node_decision_maker_name = 'node_decision_maker'
make_decision_freq = 10     # equal to send msg freq



decision_maker_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/decision_maker_config.yaml'
ballistic_predictor_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/ballistic_params.yaml'



                                                    
                                                    # Node Com
node_com_name = 'node_com'
send_msg_freq = 100
recv_from_ele_sys_freq = 100 # equal to publish topic_electric_sys_state freq


port_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/port_config.yaml'









                                                    # Node Marker 


node_marker_name = 'node_marker'