
from . import *
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


class MyNode(Node):

    def __init__(self,
                 name,
                 topic:dict):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            topic['type'],
            topic['name'],
            self.listener_callback,
            topic['qos_profile'])
        

    def listener_callback(self, msg:TFMessage):
        for tf in msg.transforms:
            self.get_logger().info(
                f"time : {tf.header.stamp.sec} {tf.header.stamp.nanosec}\n")

def main(args=None):
    rclpy.init(args=args)
    my_node = MyNode(node_subscribe_name,topic_pos)
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
