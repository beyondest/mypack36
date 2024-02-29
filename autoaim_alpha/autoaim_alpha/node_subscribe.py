
from . import *
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage



class MyNode(Node,Custom_Context_Obj):

    def __init__(self,
                 name,
                 topic:dict,
                 frequency:Union[float,None]=None):
        super().__init__(name)

        self.subscription_ = self.create_subscription(
                                                    topic['type'],
                                                    topic['name'],
                                                    self.listener_callback,
                                                    topic['qos_profile'])
                                        

        


    def listener_callback(self, msg:TFMessage):
        for tf in msg.transforms:
            
            log_info = f"\n{tf.header.frame_id}\n\
                    Target {tf.child_frame_id}\n\
                    Pos {tf.transform.translation.x}, {tf.transform.translation.y}, {tf.transform.translation.z}\n\
                    Rot {tf.transform.rotation.x}, {tf.transform.rotation.y}, {tf.transform.rotation.z}\n\
                    Time: {tf.header.stamp.sec}, {tf.header.stamp.nanosec}\n"
            self.get_logger().info(log_info)
        
    


    
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"Node {self.get_name()} get error {exc_value}")
        
        

def main(args=None):
    rclpy.init(args=args)
    
    my_node = MyNode(node_subscribe_name,topic_pos)
    
    with Custome_Context(node_subscribe_name,my_node):
        rclpy.spin(my_node)
        
    
    rclpy.shutdown()
