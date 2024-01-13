
from . import *
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

from .img import img_operation as imo

class Node_Img_Processer(Node):
    
    def __init__(self,name):
        
        super().__init__(name)
        self.sub = self.create_subscription(Image,topic_img_raw,qos_profile=10,callback=self.sub_callback)
        self.cv_bridge = CvBridge()
        self.pre_time = time.perf_counter()
        self.fps = 0
        
        
    def process(self,img:np.ndarray):
        
        if img is not None:
        
            self.get_logger().info('Receive img success')
            imo.pre_process3
            imo.add_text(img,'FPS',self.fps,scale_size=1)
            
            cv2.imshow('h',img)
            cur_time = time.perf_counter()
            self.fps = 1/(cur_time-self.pre_time)
            self.pre_time = cur_time
            
            cv2.waitKey(1)
        else:
            self.get_logger().info("Receive None")
        
    
    
    def sub_callback(self,data):
        img = self.cv_bridge.imgmsg_to_cv2(data,img_type)
        self.process(img)
        
    def _start(self):
        cv2.namedWindow('h',cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Node Img Processer start success")
    
    def _end(self):
        cv2.destroyAllWindows()
        self.get_logger().info(f"Node Img Processer end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"get error {exc_value}")
    
    
def main(args = None):
    
    rclpy.init(args=args)
    node = Node_Img_Processer(node_img_processer)
    with Custome_Context(node_img_processer,node):
        rclpy.spin(node)
    rclpy.shutdown()
        
        

















