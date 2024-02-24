
from . import *
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from .img.detector import Armor_Detector


class Node_Detect(Node,Custom_Context_Obj):
    
    def __init__(self,
                 name,
                 topic:dict):
        
        super().__init__(name)
        
        
        self.sub = self.create_subscription(
                                            topic['type'],
                                            topic=topic['name'],
                                            qos_profile=topic['qos_profile'],
                                            callback=self.sub_callback
                                            )
        
        
        self.cv_bridge = CvBridge()
        self.armor_detector = Armor_Detector(
                                            armor_color=armor_color,
                                            mode=mode,
                                            tradition_config_folder=tradition_config_folder,
                                            net_config_folder=net_config_folder
                                             )
        
        self.pre_time = time.perf_counter()
        self.fps = 0
        self.window_name = 'result'
        
        
    def find_target(self,img:np.ndarray):
        
        result , find_time = self.armor_detector.get_result(img,img)
        self.cur_time = time.perf_counter()
        self.fps = round(1/(self.cur_time-self.pre_time))
        self.pre_time = self.cur_time
        
        if mode == 'Dbg':
            self.armor_detector.visualize(img,fps=self.fps)
        
        
        self.get_logger().info(f"find target time:{find_time}s, fps:{self.fps:.2f}")
        self.get_logger().info(f"result:{result}")

        
    def sub_callback(self,data):
        
        img = self.cv_bridge.imgmsg_to_cv2(data,camera_output_format)
        self.find_target(img)
        
        
        
    def _start(self):
        
        cv2.namedWindow(self.window_name,cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        cv2.destroyAllWindows()
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"Node {self.get_name()} get error {exc_value}")
        
        
    
    
def main(args = None):
    
    rclpy.init(args=args)
    node = Node_Detect(node_detect_name,topic_img_raw)
    with Custome_Context(node_detect_name,node):
        rclpy.spin(node)
    rclpy.shutdown()
        
        

















