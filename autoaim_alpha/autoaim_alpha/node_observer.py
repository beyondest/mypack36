from . import *
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


from .decision_maker.decision_maker import *
from .decision_maker.observer import *
from .decision_maker.ballistic_predictor import *
from .port_slavedevice.port import *


class Node_Observer(Node,Custom_Context_Obj):

    def __init__(self,
                 name,
                ):
        super().__init__(name)

        self.sub = self.create_subscription(
                                            topic_detect_result['type'],
                                            topic_detect_result['name'],
                                            self.detect_sub_callback,
                                            topic_detect_result['qos_profile'])
        

        
        self.pub_armor_pos_without_correct = self.create_publisher(topic_armor_pos_without_correct['type'],
                                                                    topic_armor_pos_without_correct['name'],
                                                                    topic_armor_pos_without_correct['qos_profile'])
        
        self.pub_armor_pos_corrected = self.create_publisher(topic_armor_pos_corrected['type'],
                                                             topic_armor_pos_corrected['name'],
                                                             topic_armor_pos_corrected['qos_profile'])
        
        self.pub_armor_pos_predicted = self.create_publisher(topic_armor_pos_predicted['type'],
                                                           topic_armor_pos_predicted['name'],
                                                           topic_armor_pos_predicted['qos_profile'])
        
        self.pub_car_pos = self.create_publisher(topic_car_pos['type'],
                                                 topic_car_pos['name'],
                                                 topic_car_pos['qos_profile'])
        
        
        self.timer_correct = self.create_timer(1 / observer_correct_freq, self.timer_correct_callback)
        self.timer_predict = self.create_timer(1 / observer_predict_freq, self.timer_predict_callback)

        self.observer =  Observer(mode,observer_config_yaml_path)                 
        
        
        
    def detect_sub_callback(self, msg:DetectResult):
        
        all_target_list = []
        for each_detect_result in msg.detect_result:
            
            armor_name = each_detect_result.armor_name
            tvec = np.array([each_detect_result.pose.pose.position.x,
                             each_detect_result.pose.pose.position.y,
                             each_detect_result.pose.pose.position.z])
            q = Quaternion(each_detect_result.pose.pose.orientation.w,
                           each_detect_result.pose.pose.orientation.x,
                           each_detect_result.pose.pose.orientation.y,
                           each_detect_result.pose.pose.orientation.z)
            rvec = q.get_axis() * q.angle
            t = each_detect_result.pose.header.stamp.sec + each_detect_result.pose.header.stamp.nanosec * 1e-9
            all_target_list.append({'armor_name':armor_name,'tvec':tvec,'rvec':rvec,'time':t})
           
            
        self.observer.update_by_detection_list(all_target_list)
        self.get_logger().info(f"Observer update detection list")
        
        if if_pub_armor_state_without_correct:
            armor_state_list = self.observer.get_armor_latest_state(if_correct_state=False)
            self.publis_armor_state(self.pub_armor_pos_without_correct,armor_state_list)
            self.get_logger().info(f"publish armor state without correct")
        
    def timer_correct_callback(self):
        
        self.observer.update_by_correct_all()
        self.get_logger().info(f"observer update predition all")
        
        if if_pub_armor_state_corrected:
            armor_state_list = self.observer.get_armor_latest_state()
            self.publis_armor_state(self.pub_armor_pos_corrected,armor_state_list)
            self.get_logger().info(f"publish armor state corrected")
        
        
        if if_pub_car_state:
            
            car_state_list = self.observer.get_car_latest_state()
            self.publish_car_state(self.pub_car_pos,car_state_list)
            self.get_logger().info(f"publish car state")      
                
            
        if mode == 'Dbg':
            
            log_info = f"\nCar State:\n{car_state_list}\n\nArmor State:\n{armor_state_list}\n"
            self.get_logger().debug(log_info)
    
    def timer_predict_callback(self):
        
        if if_pub_armor_state_predicted:
            armor_state_list = self.observer.get_armor_latest_state(if_correct_state=True)
            for armor_state in armor_state_list:
                predict_time = armor_state['armor_time'] + predict_time_offset 
                
                tvec,rvec = self.observer.predict_armor_state_by_itself(armor_state['armor_name'],
                                                                        armor_state['armor_id'],
                                                                        predict_time
                                                                        )
                armor_state['armor_tvec'] = tvec
                armor_state['armor_rvec'] = rvec
                armor_state['armor_time'] = predict_time
                
            self.publis_armor_state(self.pub_armor_pos_predicted,armor_state_list)
            self.get_logger().info(f"publish armor state predicted")
            
        else:
            pass
            

    
    def publis_armor_state(self,pub_listher,armor_state_list:list):
        
        for armor_state in armor_state_list:
            msg = ArmorPos()
            msg.armor_name = armor_state['armor_name']
            msg.confidence = armor_state['armor_confidence']
            msg.pose.pose.position.x = armor_state['armor_tvec'][0]
            msg.pose.pose.position.y = armor_state['armor_tvec'][1]
            msg.pose.pose.position.z = armor_state['armor_tvec'][2]
            q = Quaternion(armor_state['armor_rvec'],np.linalg.norm(armor_state['armor_rvec']))
            msg.pose.pose.orientation.w = q.w
            msg.pose.pose.orientation.x = q.x
            msg.pose.pose.orientation.y = q.y
            msg.pose.pose.orientation.z = q.z
            msg.pose.header.stamp.sec = int(armor_state['armor_time'])
            msg.pose.header.stamp.nanosec = int((armor_state['armor_time'] - int(armor_state['armor_time'])) * 1e9)
            pub_listher.publish(msg)
        
    def publish_car_state(self,publisher,car_state_list:list):
        for car_state in car_state_list:
                
            msg = CarPos()
            msg.armor_name = car_state['armor_name']
            msg.confidence = car_state['confidence']
            msg.pose.pose.position.x = car_state['car_center_tvec'][0]
            msg.pose.pose.position.y = car_state['car_center_tvec'][1]
            msg.pose.pose.position.z = car_state['car_center_tvec'][2]
            q = Quaternion(car_state['car_center_rvec'],np.linalg.norm(car_state['car_center_tvec']))
            msg.pose.pose.orientation.w = q.w
            msg.pose.pose.orientation.x = q.x
            msg.pose.pose.orientation.y = q.y
            msg.pose.pose.orientation.z = q.z
            msg.pose.header.stamp.sec = int(car_state['car_time'])
            msg.pose.header.stamp.nanosec = int((car_state['car_time'] - int(car_state['car_time'])) * 1e9)
            
            msg.twist.twist.linear.x = car_state['tv_vec'][0]
            msg.twist.twist.linear.y = car_state['tv_vec'][1]
            msg.twist.twist.linear.z = car_state['tv_vec'][2]
            rv_vec = car_rotation_axis * car_state['rotation_speed']
            msg.twist.twist.angular.x = rv_vec[0]
            msg.twist.twist.angular.y = rv_vec[1]
            msg.twist.twist.angular.z = rv_vec[2]
            publisher.publish(msg)
                
    
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"Node {self.get_name()} get error {exc_value}")
        

def main(args=None):
    rclpy.init(args=args)
    my_node = Node_Observer(node_decision_maker_name)
    
    with Custome_Context(node_observer_name,my_node):
        rclpy.spin(my_node)
        
    rclpy.shutdown()
