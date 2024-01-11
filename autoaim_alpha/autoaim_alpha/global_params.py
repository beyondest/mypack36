import sys
sys.path.append('../..')


import utils_network
import img.img_operation as imo
import os_op
import camera
import rclpy.qos


from os_op.global_logger import *
from os_op.basic import *


topic_img_raw = 'img_raw'
img_type = 'bgr8'
qos_profile_img_raw = rclpy.qos.QoSProfile(
    history = rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
    depth = 10,
    reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
    durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE
)
        



