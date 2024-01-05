from typing import List
from mypath import *

import rclpy
import cv2
import cv_bridge
from rclpy.context import Context
from rclpy.parameter import Parameter
import rclpy.qos
from sensor_msgs.msg import Image
import numpy as np


import camera.control as cac

class Mindvision_Webcam(rclpy.Node):
    def __init__(self,
                 name:str,
                 timer_s:float
                 ):
        super().__init__(name)
        
        qos_profile = rclpy.qos.QoSProfile(
            history = rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth = 10,
            reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE
        )
        
        self.publisher = self.create_publisher(Image,"image_raw",qos_profile)
        self.timer = self.create_timer(timer_s)
        self.cv_bridge = cv_bridge.CvBridge()
        
        
        

