
from . import *
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from .img.detector import Armor_Detector
from visualization_msgs.msg import Marker
from pyquaternion import Quaternion



class Node_Detector(Node,Custom_Context_Obj):
    
    def __init__(self,
                 name,
                 ):
        
        super().__init__(name)
        self.sub_img_raw = self.create_subscription(
                                            topic_img_raw['type'],
                                            topic_img_raw['name'],
                                            self.sub_img_raw_callback,
                                            topic_img_raw['qos_profile']
                                            )
        self.pub_detect_result = self.create_publisher(
                                            topic_detect_result['type'],
                                            topic_detect_result['name'],
                                            topic_detect_result['qos_profile']
                                            )
        self.pub_img_detected = self.create_publisher(topic_img_detected['type'],
                                             topic_img_detected['name'],
                                             topic_img_detected['qos_profile'])

        
        
        self.cv_bridge = CvBridge()
        self.armor_detector = Armor_Detector(
                                            armor_color=armor_color,
                                            mode=mode,
                                            tradition_config_folder=tradition_config_folder,
                                            net_config_folder=net_config_folder,
                                            depth_estimator_config_yaml=depth_estimator_config_yaml_path
                                             )
        
        self.pre_time = time.perf_counter()
        self.fps = 0
        self.window_name = 'result'
        if mode == 'Dbg':
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
         
    def sub_img_raw_callback(self,data):
        
        img = self.cv_bridge.imgmsg_to_cv2(data,camera_output_format)
        
        result , find_time = self.armor_detector.get_result(img,img)
        self.cur_time = time.perf_counter()
        self.fps = round(1/(self.cur_time-self.pre_time))
        self.pre_time = self.cur_time
        
        if if_pub_img_detected:
            visualize_result = self.armor_detector.visualize(img,fps=self.fps,windows_name=None)
            self.pub_img_detected.publish(self.cv_bridge.cv2_to_imgmsg(visualize_result,camera_output_format))
            self.get_logger().debug(f"publish visualize result success")
            
        
        if result is not None:
            msg = DetectResult()
            t = self.get_clock().now().to_msg()
            for each_result in result:
                ed = EachDetectResult()
                
                ed.armor_name = each_result['name']
                ed.confidence = each_result['probability']
                ed.pose.header.stamp = t
                ed.pose.header.frame_id = 'camere_frame'
                ed.pose.pose.position.x = each_result['pos'][0]
                ed.pose.pose.position.y = each_result['pos'][1]
                ed.pose.pose.position.z = each_result['pos'][2]
                
                q = Quaternion(axis=each_result['rvec'],radians=np.linalg.norm(each_result['rvec']))
                ed.pose.pose.orientation.w = q.w
                ed.pose.pose.orientation.x = q.x
                ed.pose.pose.orientation.y = q.y
                ed.pose.pose.orientation.z = q.z
                
                msg.detect_result.append(ed)
                
                log_info = f"armor_name:{ed.armor_name},confidence:{ed.confidence},pos:{ed.pose.pose.position},orientation:{ed.pose.pose.orientation} time:{t.sec}s{t.nanosec/1000000}ns"
                
                self.get_logger().debug(f"Find target : {log_info} spend time:{find_time}s")
                    
            self.pub_img_detected.publish(msg)
            self.get_logger().debug(f"publish detect result success")
        else:
            self.get_logger().debug(f"no target found")
        
    def _start(self):
        
        cv2.namedWindow(self.window_name,cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        cv2.destroyAllWindows()
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        self.get_logger().error(f"Node {self.get_name()} get error {exc_value}")
        
        
def main(args = None):
    
    rclpy.init(args=args)
    node = Node_Detector(node_detector_name)
    with Custome_Context(node_detector_name,node):
        rclpy.spin(node)
    rclpy.shutdown()
        
        

















