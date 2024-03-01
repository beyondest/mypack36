
from . import *
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from .img.detector import Armor_Detector
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

class Node_Detector(Node,Custom_Context_Obj):
    
    def __init__(self,
                 name,
                 topic:dict,
                 topic2:dict):
        
        super().__init__(name)
        
        
        self.sub = self.create_subscription(
                                            topic['type'],
                                            topic=topic['name'],
                                            qos_profile=topic['qos_profile'],
                                            callback=self.sub_callback
                                            )
        self.pub = self.create_publisher(
                                            topic2['type'],
                                            topic=topic2['name'],
                                            qos_profile=topic2['qos_profile']
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
        self.get_logger().info(f'find result:{result}')
        
        if result is not None:
            msg = TFMessage()
            for each_result in result:
                tf = TransformStamped()
                
                tf.header.frame_id = 'Detector_0:'
                tf.header.stamp = self.get_clock().now().to_msg()

                tf.child_frame_id = each_result['name']
                tf.transform.translation.x = each_result['pos'][0]
                tf.transform.translation.y = each_result['pos'][1]
                tf.transform.translation.z = each_result['pos'][2]
                tf.transform.rotation.x = each_result['rvec'][0]
                tf.transform.rotation.y = each_result['rvec'][1]    
                tf.transform.rotation.z = each_result['rvec'][2]
                msg.transforms.append(tf)
                
                
                log_info = f"\n{tf.header.frame_id}\n\
                        Target {tf.child_frame_id}\n\
                        Pos {tf.transform.translation.x}, {tf.transform.translation.y}, {tf.transform.translation.z}\n\
                        Rot {tf.transform.rotation.x}, {tf.transform.rotation.y}, {tf.transform.rotation.z}\n\
                        Time: {tf.header.stamp.sec}, {tf.header.stamp.nanosec}\n"
                self.get_logger().debug(log_info)
                    
            self.pub.publish(msg)
            self.get_logger().info(f"publish tf success")
        else:
            self.get_logger().info(f"not publish tf")
            
        
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
    node = Node_Detector(node_detector_name,
                         topic_img_raw,
                         topic_armor_pos)
    with Custome_Context(node_detector_name,node):
        rclpy.spin(node)
    rclpy.shutdown()
        
        

















