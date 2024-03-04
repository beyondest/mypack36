import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped,TransformStamped,PointStamped
from tf2_msgs.msg import TFMessage
import math
from autoaim_interface.msg import ElectricsysState
class PosePublisher(Node):

    def __init__(self):
        super().__init__('pose_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'pose', 10)
        self.pub2 = self.create_publisher(ElectricsysState, 'test', 10)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.time = 0.0

    def timer_callback(self):

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = math.cos(self.time) * 2.0
        msg.pose.position.y = math.sin(self.time) * 2.0
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(self.time / 2.0 * 10)
        msg.pose.orientation.w = math.cos(self.time / 2.0 * 10)
    
        
        
        self.publisher_.publish(msg)
        self.time += 0.1
        self.get_logger().info('Publishing: "%s"' % msg)
        
        msg2 = ElectricsysState()
        msg2.cur_pitch = 0.1
        
        self.pub2.publish(msg2)
        self.get_logger().info('Publishing: "%s"' % msg2)
        
        
def main(args=None):
    rclpy.init(args=args)
    pose_publisher = PosePublisher()
    rclpy.spin(pose_publisher)
    pose_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
