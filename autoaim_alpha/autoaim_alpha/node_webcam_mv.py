from . import *

from rclpy.node import Node
import cv_bridge
from rclpy.context import Context
from rclpy.parameter import Parameter
import rclpy.qos
from sensor_msgs.msg import Image

from .camera.mv_class import *



class Node_Webcam_MV(Node,Custom_Context_Obj):
    
    def __init__(self,
                 name:str,
                 topic:dict,
                 frequency:float
                 ):
        
        super().__init__(name)
        

        self.publisher = self.create_publisher(
                                               topic['type'],
                                               topic=topic['name'],
                                               qos_profile=topic['qos_profile']
                                               )
        
        self.timer_pub_img = self.create_timer(1/frequency,self.timer_pub_img_callback)
        self.cv_bridge = cv_bridge.CvBridge()
        
        self.mv = Mindvision_Camera(
                                    output_format = camera_output_format,
                                    camera_mode=node_webcam_mv_mode,
                                    camera_config_folder = camera_config_folder,
                                    armor_color=armor_color,
                                    if_yolov5=if_yolov5
                                    )
        
        
        #self.mv.enable_trackbar_config()
        
        
        if node_webcam_mv_mode == 'Dbg':
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        
    def timer_pub_img_callback(self):
        
        img = self.mv.get_img()
        
        if img is not None:
            
            self.publisher.publish(self.cv_bridge.cv2_to_imgmsg(img,camera_output_format))
            
        else:
            self.get_logger().error("No img get from camera")
            
    
    def _start(self):
        
        self.mv._start()
        self.get_logger().info(f"Node {self.get_name()} start success")
        
    def _end(self):
        
        self.mv._end()
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()
        

    def _errorhandler(self,exc_value):

        self.get_logger().error(f"Node {self.get_name()} get error {exc_value}")
        
def main(args = None):
    
    rclpy.init(args=args)
    
    node = Node_Webcam_MV(node_webcam_mv_name,
                          topic_img_raw,
                          node_webcam_mv_frequency
                          )
    
    with Custome_Context(node_webcam_mv_name,node,[KeyboardInterrupt]):
        rclpy.spin(node)
        
    rclpy.shutdown()
    
        
        
    
    
        
        

