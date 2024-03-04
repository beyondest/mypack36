import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from autoaim_interface.msg import ElectricsysState

class MarkerPublisher(Node):

    def __init__(self):
        super().__init__('marker_publisher')
        self.publisher_ = self.create_publisher(Marker, 'visualization_marker', 10)
        self.subscription = self.create_subscription(PoseStamped, 'pose', self.listener_callback, 10)
        self.sub2 = self.create_subscription(ElectricsysState, 'test', self.listener_callback2, 10)
        
        

    def listener_callback(self, msg: PoseStamped):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "basic_shapes"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = msg.pose
        
        
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.publisher_.publish(marker)
        self.get_logger().info('Marker published')
    
    
    def listener_callback2(self, msg: ElectricsysState):
        self.get_logger().info('Electricity state received: %s' % msg)
        

def main(args=None):
    
    rclpy.init(args=args)
    marker_publisher = MarkerPublisher()
    rclpy.spin(marker_publisher)
    marker_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
