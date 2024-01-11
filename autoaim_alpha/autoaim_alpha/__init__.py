import rclpy.qos
from . import camera
from . import os_op
from . import img
topic_img_raw = 'img_raw'
img_type = 'bgr8'
node_webcam_mv = 'node_webcam_mv'
node_img_processer = 'node_img_processer'

qos_profile_img_raw = rclpy.qos.QoSProfile(
    history = rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
    depth = 10,
    reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
    durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE
)
        



