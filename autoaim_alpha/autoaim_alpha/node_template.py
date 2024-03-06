from . import *
from .os_op.basic import *
import rclpy
from rclpy.node import Node



class MyNode(Node,Custom_Context_Obj):

    def __init__(self,
                 name,
                 topic:dict,
                 frequency:Union[float,None] = None):
        super().__init__(name)
        self.publisher_ = self.create_publisher(topic['type'],
                                                topic['name'],
                                                topic['qos_profile'])
        
        self.subscription_ = self.create_subscription(topic['type'],
                                                      topic['name'],
                                                      self.listener_callback,
                                                      topic['qos_profile'])
                                        

        

        self.timer = self.create_timer(1/frequency, self.timer_callback)



    def listener_callback(self, msg:TFMessage):
        for tf in msg.transforms:
            self.get_logger().info(tf)




    def timer_callback(self):
        msg = ...

        self.publisher_.publish(msg)
        self.get_logger().info(f"Published msg: {msg}")
    
    
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        self.get_logger().error(f"Node {self.get_name()} get error {exc_value}")
        
        

def main(args=None):
    rclpy.init(args=args)
    
    my_node = MyNode(node_publish_name,topic_pos,node_publish_frequency)
    
    with Custome_Context(node_publish_name,my_node):
        rclpy.spin(my_node)
        
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
