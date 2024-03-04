from . import *

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


id_to_color_dict = {0 : (1,0,0),1 : (0,1,0),2 : (0,0,1),3 : (1,1,0)}
class Armor_Marker:
    def __init__(self,armor_name,armor_id):
        
        self.armor_name = armor_name
        self.armor_id = armor_id
        self.marker = Marker()
        self.marker.header.frame_id = "map"
        
        self.marker.ns = "basic_shapes"
        self.marker.id = 0
        self.marker.type = Marker.CUBE
        self.marker.action = Marker.ADD
        
        self.marker.scale.x = 0.5
        self.marker.scale.y = 0.5
        self.marker.scale.z = 0.2
        color = id_to_color_dict[self.armor_id]
        self.marker.color.r = color[0]
        self.marker.color.g = color[1]
        self.marker.color.b = color[2]
        
        self.marker.color.a = 0

class Node_Marker(Node,Custom_Context_Obj):

    def __init__(self,name):
        super().__init__(name)
        self.publisher_ = self.create_publisher(topic_armor_pos_corrected['type'], 
                                                 topic_armor_pos_corrected['name'],
                                                 topic_armor_pos_corrected['qos_profile'])
        
        
        self.subscription = self.create_subscription(topic_armor_pos_without_correct['type'], 
                                                     topic_armor_pos_without_correct['name'],
                                                     self.listener_callback,
                                                     topic_armor_pos_without_correct['qos_profile']
                                                     )
        self.enemy_car_list =enemy_car_list
        
        
        self.marker_list = []
        
        for enemy_car in self.enemy_car_list:
            for i in range(enemy_car['armor_nums']):
                armor_marker = Armor_Marker(enemy_car['armor_name'],i)
                armor_marker.marker.header.stamp = self.get_clock().now().to_msg()
                self.marker_list.append(armor_marker)
        
        
    def listener_callback(self, msg:ArmorPos):
        for armor_marker in self.marker_list:
            if armor_marker.armor_name == msg.armor_name and armor_marker.armor_id == msg.armor_id:
                armor_marker.marker.pose = msg.pose.pose
                armor_marker.marker.header.stamp = msg.pose.header.stamp
                self.publisher_.publish(armor_marker.marker)
                
                self.get_logger().info(f"Publish marker {armor_marker.armor_name} {armor_marker.armor_id}")
                
       
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"Node {self.get_name()} get error {exc_value}")

def main(args=None):
    
    rclpy.init(args=args)
    node = Node_Marker(node_marker_name)
    
    with Custome_Context(node_marker_name,node):
        rclpy.spin(node)
        
    rclpy.shutdown()

if __name__ == '__main__':
    main()
