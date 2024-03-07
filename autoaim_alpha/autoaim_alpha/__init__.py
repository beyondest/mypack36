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

general_config = Data.get_file_info_from_yaml('/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/general_params.yaml')
mode = general_config['mode']
armor_color = general_config['armor_color']

enemy_car_list=  general_config['enemy_car_list']
armor_name_to_init_state = general_config['armor_name_to_init_state']

    
    
                                                    # Public Params
car_rotation_axis = np.array([0,1,0])
# detect_delay + correct_and_predict_delay + cal_ballistic_delay + com_delay
total_delay = 0.03 + 0.01 + 0.01 + 0.005
predict_time_offset = 0.3

topic_img_raw = {'name': 'img_raw', 'type': Image, 'qos_profile':3}
topic_electric_sys_state = {'name': 'electric_sys_state', 'type':ElectricsysState , 'qos_profile':10}
topic_detect_result = {'name': 'detect_result', 'type': DetectResult, 'qos_profile':10}
topic_armor_pos_without_correct = {'name': 'armor_pos_without_correct', 'type': ArmorPos, 'qos_profile':10}
topic_armor_pos_corrected = {'name': 'armor_pos_corrected', 'type': ArmorPos, 'qos_profile':10}
topic_armor_pos_predicted = {'name': 'armor_pos_predicted', 'type': ArmorPos, 'qos_profile':10}
topic_car_pos = {'name': 'car_pos', 'type': CarPos, 'qos_profile':10}
topic_electric_sys_com = {'name': 'electric_sys_com', 'type': ElectricsysCom, 'qos_profile':10}

topic_marker_armor_pos_corrected = {'name':'armor_marker_topic', 'type': Marker, 'qos_profile':10}
topic_marker_text_corrected = {'name':'armor_text_topic', 'type': Marker, 'qos_profile':10}



if_pub_car_state = False
if_pub_armor_state_without_correct = False
if_pub_armor_state_corrected = True
if_pub_armor_state_predicted = False

node_webcam_mv_mode = 'Rel'
node_detector_mode = 'Dbg'
node_observer_mode = 'Rel'
node_decision_maker_mode = 'Rel'
node_com_mode = 'Rel'
node_marker_mode = 'Rel'


                                                    # Node Webcam Mv
                                                    
node_webcam_mv_name = 'node_webcam_mv'
node_webcam_mv_frequency = 50

camera_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/camera_config' 


camera_output_format = 'bgr8'





                                                
                                                    # Node Detector


node_detector_name = 'node_detect'
tradition_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/tradition_config'
net_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/net_config'
depth_estimator_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/other_config/pnp_params.yaml'



                                                    # Node Observer
                                                    
node_observer_name = 'node_observer'
observer_correct_freq = 100
observer_predict_freq = 100

observer_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/other_config/observer_params.yaml'


                                                  
                                                    # Node Decision Maker
node_decision_maker_name = 'node_decision_maker'
make_decision_freq = 100     # equal to send msg freq



decision_maker_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/other_config/decision_maker_params.yaml'
ballistic_predictor_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/other_config/ballistic_params.yaml'



                                                    
                                                    # Node Com
node_com_name = 'node_com'
send_msg_freq = 100
recv_from_ele_sys_freq = 100 # equal to publish topic_electric_sys_state freq


port_config_yaml_path = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/autoaim_alpha/config/other_config/port_params.yaml'









                                                    # Node Marker 


node_marker_name = 'node_marker'