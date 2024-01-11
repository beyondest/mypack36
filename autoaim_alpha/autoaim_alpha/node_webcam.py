from typing import List
from autoaim_alpha.autoaim_alpha.global_params import *

import rclpy
import cv_bridge
from rclpy.context import Context
from rclpy.parameter import Parameter
import rclpy.qos
from sensor_msgs.msg import Image
from camera.mv_class import Mindvision_Camera




class Node_Webcam_MV(rclpy.Node):
    def __init__(self,
                 name:str,
                 time_s:float
                 ):
        super().__init__(name)
        

        self.publisher = self.create_publisher(Image,topic_img_raw,qos_profile=qos_profile_img_raw)
        self.timer_pub_img = self.create_timer(time_s,)
        self.cv_bridge = cv_bridge.CvBridge()
        self.mv = Mindvision_Camera()
        
        
    def timer_pub_img_callback(self):
        img = self.mv.get_img_continous()
        if img is not None:
            self.publisher.publish(self.cv_bridge.cv2_to_imgmsg(img,img_type))
        else:
            self.get_logger().error("No img get from camera")
            
        self.get_logger().info("Publish img success")
    
    def _start(self):
        self.mv._start()
        self.get_logger().info(f"Node {self.get_name()} start success")
        
    def _end(self):
        
        self.mv._end()
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()
        

    def _errorhandler(self,exc_value):
        print(f"Get error : {exc_value}")
    
def main(args = None):
    
    rclpy.init(args=args)
    
    node = Node_Webcam_MV('MV1',0.01)
    
    with Custome_Context('Webcam',node):
        rclpy.spin(node)
        
    rclpy.shutdown()
    
        
        
    
    
        
        

