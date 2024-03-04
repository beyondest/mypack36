import rclpy
from rclpy.node import Node
import time


class nodetest3(Node):
    def __init__(self):
        super().__init__('nodetest3')
        self.i = 0
        t1 =self.create_timer(0.5, self.timer_callback)
        t2 = self.create_timer(1.0, self.t2_callback)
        
    def timer_callback(self):
        self.get_logger().info('Hello World: %d' % self.i)
        
        
    def t2_callback(self):
        self.get_logger().info('Hello World2: %d' % self.i)
        self.i += 1
    



def main(args=None):
    rclpy.init(args=args)
    node = nodetest3()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()