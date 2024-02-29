import rclpy.qos
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from . import os_op
from .os_op.basic import *
from .utils_network.data import Data




armor_type_list = ['1d','2x','3d','3x','4d','4x','5d','5x','based','basex','sentry']


general_config = Data.get_file_info_from_yaml('/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/general_config.yaml')
mode = general_config['mode']
armor_color = general_config['armor_color']

                                                    # Public Params


topic_img_raw = {'name': 'img_raw', 'type': Image, 'qos_profile':3}

topic_pos = {'name': 'pos', 'type': TFMessage, 'qos_profile':10}


                                                    # Node webcam mv
                                                    
node_webcam_mv_name = 'node_webcam_mv'
node_webcam_mv_frequency = 30


camera_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/camera_config' 
tradition_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/tmp_tradition_config'


camera_output_format = 'bgr8'





                                                
                                                    # Node detect


node_detect_name = 'node_detect'

tradition_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/tmp_tradition_config'
net_config_folder = '/home/liyuxuan/vscode/pywork_linux/autoaim_ws/src/mypack36/tmp_net_config'


                                                    # Node publish
                                                    
                                                    
node_publish_name = 'node_publish'
node_publish_frequency = 10


                                                    # Node subscribe
                                                    
node_subscribe_name = 'node_subscribe'



        



