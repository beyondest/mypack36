from . import *

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

group_1_id_to_color_dict = {0 : (1.0 ,0.0 ,0.0),
                            1 : (1.0 ,0.5 ,0.0),
                            2 : (1.0 ,0.8 ,0.0),
                            3 : (1,0 ,1.0 ,0.0)}


group_2_id_to_color_dict = {0 : (0.0 ,1.0 ,0.0),
                            1 : (0.0 ,1.0 ,0.5),
                            2 : (0.0 ,1.0 ,0.8),
                            3 : (0.0 ,1.0 ,1.0)}

group_3_id_to_color_dict = {0 : (0.0 ,0.0 ,1.0),
                            1 : (0.5 ,0.0 ,1.0),
                            2 : (0.8 ,0.0 ,1.0),
                            3 : (1.0 ,0.0 ,1.0)}




id_to_color_dict = {0 : (1,0,0),1 : (0,1,0),2 : (0,0,1),3 : (1,1,0)}

small_armor_wid = 0.135
small_armor_hei = 0.125
small_armor_thickness = 0.02
big_armor_wid = 0.23
big_armor_hei = 0.127
big_armor_thickness = 0.02

def marker_id_generator(max_id:int):
    for i in range(max_id):
        yield i

marker_id_generator_obj = marker_id_generator(100)

class Armor_Marker:
    def __init__(self,armor_name,armor_id,color_group:int = 0):
        
        self.armor_name = armor_name
        if armor_name in ['1x','2x','3x','4x','5x','sentry','basex']:
            armor_type = 'small'
        
        else:
            armor_type = 'big'
            
            
        self.armor_id = armor_id
        self.marker = Marker()
        self.marker.header.frame_id = "map"
        
        self.marker.id = next(marker_id_generator_obj)
        self.marker.type = Marker.CUBE
        
        self.marker.action = Marker.ADD
        
        self.marker.scale.x = small_armor_wid if armor_type =='small' else big_armor_wid
        self.marker.scale.y = small_armor_thickness if armor_type =='small' else big_armor_thickness
        self.marker.scale.z = small_armor_hei if armor_type =='small' else big_armor_hei
        
        if color_group == 1:
            self.marker.ns = "group_1"
            color = group_1_id_to_color_dict[self.armor_id]
        elif color_group == 2:
            self.marker.ns = "group_2"
            color = group_2_id_to_color_dict[self.armor_id]
            
        elif color_group == 3:
            self.marker.ns = "group_3"
            color = group_3_id_to_color_dict[self.armor_id]
        else:
            color = id_to_color_dict[self.armor_id]
            
        self.marker.color.r = float(color[0])
        self.marker.color.g = float(color[1])
        self.marker.color.b = float(color[2])
        
        #transparent of marker
        self.marker.color.a = 0.2

class Camera_Marker:
    def __init__(self):
        
        self.marker = Marker() 
        self.marker.header.frame_id = "map"
        self.marker.id = next(marker_id_generator_obj)
        self.marker.type = Marker.CUBE
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.2
        self.marker.scale.y = 0.2
        self.marker.scale.z = 0.2
        self.marker.color.r = 1.0
        self.marker.color.g = 1.0
        self.marker.color.b = 1.0
        self.marker.color.a = 0.2
        self.marker.ns = "camera"
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0    
        
        


class Node_Marker(Node,Custom_Context_Obj):

    def __init__(self,name):
        super().__init__(name)
        
        self.pub_armor_marker = self.create_publisher(topic_marker_armor_pos_corrected['type'], 
                                                 topic_marker_armor_pos_corrected['name'],
                                                 topic_marker_armor_pos_corrected['qos_profile'])
        
        self.pub_text_marker = self.create_publisher(topic_marker_text_corrected['type'], 
                                           topic_marker_text_corrected['name'],
                                           topic_marker_text_corrected['qos_profile'])
        
        self.enemy_car_list =enemy_car_list
        
        
        if if_pub_armor_state_corrected:
            self.corrected_marker_list = []
            self._init_marker_list(self.corrected_marker_list,1)
            self.sub_corrected_ = self.create_subscription(topic_armor_pos_corrected['type'], 
                                                        topic_armor_pos_corrected['name'],
                                                        self.armor_pos_corrected_listener,
                                                        topic_armor_pos_corrected['qos_profile']
                                                        )
            
        if if_pub_armor_state_without_correct:
            self.without_corrected_marker_list = []
            self._init_marker_list(self.without_corrected_marker_list,2)
            self.sub_without_correct_ = self.create_subscription(topic_armor_pos_without_correct['type'], 
                                                        topic_armor_pos_without_correct['name'],
                                                        self.armor_pos_without_corrected_listener,
                                                        topic_armor_pos_without_correct['qos_profile']
                                                        )
            
            
        if if_pub_armor_state_predicted:
            self.predicted_marker_list = []
            self._init_marker_list(self.predicted_marker_list,3)
            self.sub_predicted_ = self.create_subscription(topic_armor_pos_predicted['type'], 
                                                        topic_armor_pos_predicted['name'],
                                                        self.armor_pos_predicted_listener,
                                                        topic_armor_pos_predicted['qos_profile']
                                                        )
            
        
        
            
        if node_marker_mode == 'Dbg':
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

    def armor_pos_corrected_listener(self, msg:ArmorPos):
        self._update_marker_list(self.corrected_marker_list,msg)
                
    def armor_pos_without_corrected_listener(self, msg:ArmorPos):
        self._update_marker_list(self.without_corrected_marker_list,msg)
                
    def armor_pos_predicted_listener(self, msg:ArmorPos):
        self._update_marker_list(self.predicted_marker_list,msg)
    
    def _init_marker_list(self,marker_list:list,color_group:int = 0):    
        for enemy_car in self.enemy_car_list:
            for i in range(enemy_car['armor_nums']):
                armor_marker = Armor_Marker(enemy_car['armor_name'],i,color_group)
                armor_marker.marker.header.stamp = self.get_clock().now().to_msg()
                marker_list.append(armor_marker) 
                
    def _update_marker_list(self,marker_list:list,msg:ArmorPos):
        
        for armor_marker in marker_list:
            
            if armor_marker.armor_name == msg.armor_name and armor_marker.armor_id == msg.armor_id:
                armor_marker.marker.pose = msg.pose.pose
                armor_marker.marker.header.stamp = msg.pose.header.stamp
                self.pub_armor_marker.publish(armor_marker.marker)
                self._add_text_to_marker(armor_marker.marker,f"{armor_marker.armor_name}:{armor_marker.armor_id}")
                self._add_text_to_marker(armor_marker.marker,
                                         f"y:{msg.pose.pose.position.y:.2f}",
                                           id_offset=200,
                                           pos_offset=np.array([0,0,0.1]))
                
    
    def _add_text_to_marker(self,
                            marker:Marker,
                            text:str,
                            id_offset:int = 100,
                            pos_offset:np.ndarray = np.zeros(3)
                            ):
        m = Marker()
        
        m.header = marker.header
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = marker.pose.position.x + pos_offset[0]
        m.pose.position.y = marker.pose.position.y + pos_offset[1]
        m.pose.position.z = marker.pose.position.z + pos_offset[2]
        m.pose.orientation = marker.pose.orientation
        m.scale.z = 0.1
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.text = text
        m.id = marker.id + id_offset
        self.pub_text_marker.publish(m)
        
        
    
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):

        self.get_logger().error(f"Node {self.get_name()} get error {exc_value}")
        
def main(args=None):
    
    rclpy.init(args=args)
    node = Node_Marker(node_marker_name)
    
    with Custome_Context(node_marker_name,node,[KeyboardInterrupt]):
        rclpy.spin(node)
        
    rclpy.shutdown()

if __name__ == '__main__':
    main()
