import sys
sys.path.append('../..')
from autoaim_alpha.autoaim_alpha import *
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from os_op.basic import *
from camera.mv_class import *


class Node_Img_Processer(Node):
    def __init__(self,name):
        super().__init__(name)
        self.sub = self.create_subscription(Image,topic_img_raw,qos_profile=qos_profile_img_raw,callback=self.sub_callback)
        self.cv_bridge = CvBridge()
        
        
    def process(self,img:np.ndarray):
        if img is not None:
            cv2.imshow('h',img)
        else:
            print('Receive None')
        
        
    
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
    node = Node_Img_Processer(node_img_processer)
    with Custome_Context(node_img_processer,node):
        rclpy.spin(node)
    rclpy.shutdown()
        
        


















