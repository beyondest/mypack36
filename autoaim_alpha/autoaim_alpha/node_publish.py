from . import *
from .os_op.basic import *
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped


class MyNode(Node,Custom_Context_Obj):

    def __init__(self,
                 name,
                 topic:dict,
                 frequency:float):
        super().__init__(name)
        self.publisher_ = self.create_publisher(topic['type'], topic['name'], topic['qos_profile'])
        self.timer = self.create_timer(1/frequency, self.timer_callback)

    def timer_callback(self):
        msg = TFMessage()
    
        for i in range(5):
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = "world"
            tf.child_frame_id = f"object_{i}"
            tf.transform.translation.x = 2.0
            tf.transform.translation.y = 3.0
            tf.transform.translation.z = 4.0
            tf.transform.rotation.x = 0.0
            tf.transform.rotation.y = 0.0
            tf.transform.rotation.z = 0.0
            tf.transform.rotation.w = 1.0
            msg.transforms.append(tf)
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published {len(msg.transforms)} transforms")
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published msg: {msg}")
    
    
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"Node {self.get_name()} get error {exc_value}")
        
        

def main(args=None):
    rclpy.init(args=args)
    
    my_node = MyNode(node_publish_name,topic_pos,node_publish_frequency)
    
    with Custome_Context(node_publish_name,my_node):
        rclpy.spin(my_node)
        
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
