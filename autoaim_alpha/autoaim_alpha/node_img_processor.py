
from autoaim_alpha.global_params import *
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class Node_Img_Processer(rclpy.Node):
    def __init__(self,name):
        super().__init__(name)
        self.sub = self.create_subscription(Image,topic_img_raw,qos_profile=qos_profile_img_raw,callback=self.sub_callback)
        self.cv_bridge = CvBridge()
        
        
    def process(self,img:np.ndarray):
        cv2.imshow('h',img)
        
        
    
    def sub_callback(self,data):
        self.get_logger().info("Receive img success")
        img = self.cv_bridge.imgmsg_to_cv2(data,img_type)
        self.process(img)
        
    def _start(self):
        cv2.namedWindow('h',cv2.WINDOW_NORMAL)
        self.get_logger().info(f"Node Img Processer start success")
    
    def _end(self):
        cv2.destroyAllWindows()
        self.get_logger().info(f"Node Img Processer end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"get error {exc_value}")
    
    
def main(args = None):
    rclpy.init(args=args)
    node = Node_Img_Processer('ReceiveMV1')
    with Custome_Context('ReceiveMV1',node):
        rclpy.spin(node)
    rclpy.shutdown()
        
        


















