import rclpy.qos

from . import os_op
from .os_op.basic import *
from .utils_network.data import Data


general_config = Data.get_file_info_from_yaml('/home/rcclub/ggbond/autoaim_ws/src/mypack36/general_config.yaml')
mode = general_config['mode']
armor_color = general_config['armor_color']

                                                    # Public Params

topic_img_raw = 'img_raw'
qos_profile_img_raw = 3


                                                    # Node webcam mv
                                                    
node_webcam_mv = 'node_webcam_mv'
node_webcam_mv_frequency = 30


isp_params_path = '/home/rcclub/ggbond/autoaim_ws/src/mypack36/tradition_config/red/custom_isp_params.yaml'
tradition_config_folder = '/home/rcclub/ggbond/autoaim_ws/src/mypack36/tmp_tradition_config'


camera_output_format = 'bgr8'





                                                
                                                    # Node detector


node_detector = 'node_detector'
node_detector_mv_frequency = None

tradition_config_folder = '/home/rcclub/ggbond/autoaim_ws/src/mypack36/tmp_tradition_config'
net_config_folder = '/home/rcclub/ggbond/autoaim_ws/src/mypack36/tmp_net_config'



        



