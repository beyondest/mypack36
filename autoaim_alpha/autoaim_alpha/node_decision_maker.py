from . import *
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


from .decision_maker.decision_maker import *
from .decision_maker.observer import *
from .decision_maker.ballistic_predictor import *
from .port_slavedevice.port import *

class Node_Decision_Maker(Node,Custom_Context_Obj):

    def __init__(self,
                 name,
                 topic:dict,
                 predict_freq:float,
                 make_decision_freq:float,
                 receive_from_eletric_sys_freq : float):
        super().__init__(name)

        self.sub = self.create_subscription(
                                            topic['type'],
                                            topic['name'],
                                            self.listener_callback,
                                            topic['qos_profile'])

        self.timer_predict = self.create_timer(1 / predict_freq, self.timer_predict_callback)
        self.timer_make_decision = self.create_timer(1 / make_decision_freq , self.timer_make_decisin_callback)
        self.timer_receive_from_eletric_sys = self.create_timer(1 / receive_from_eletric_sys_freq ,self.timer_receive_from_eletric_sys_callback )
        
        self.observer =  Observer(mode,observer_config_yaml_path)                 
        self.decision_maker = Decision_Maker(mode,decision_maker_config_yaml_path)
        self.ballistic_predictor = Ballistic_Predictor(mode,ballistic_predictor_config_yaml_path)
        self.port = Port(port_config_yaml_path,mode)
        
    def listener_callback(self, msg:TFMessage):
        all_target_list = []
        for tf in msg.transforms:
            
            armor_name = tf.child_frame_id
            tvec = np.array([tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z])
            rvec = np.array([tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z])
            armor_time = tf.header.stamp.sec + tf.header.stamp.nanosec/1e9
            
            all_target_list.append({'armor_name' : armor_name, 'tvec' : tvec, 'rvec' : rvec, 'time' : armor_time})
            
        self.observer.update_by_detection_list(all_target_list)
        self.get_logger().info(f"observer update detection list")
        
    def timer_predict_callback(self):
        
        self.observer.update_by_prediction_all()
        self.get_logger().info(f"observer update predition all")
        car_state_list = self.observer.get_car_latest_state()
        armor_state_list = self.observer.get_armor_latest_state()
        
        if mode == 'Dbg':
            
            log_info = f"\nCar State:\n{car_state_list}\n\nArmor State:\n{armor_state_list}\n"
            self.get_logger().debug(log_info)
    
    def timer_make_decisin_callback(self):
        pass
    
    def timer_receive_from_eletric_sys_callback(self):
        pass
    
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"Node {self.get_name()} get error {exc_value}")
        

def main(args=None):
    rclpy.init(args=args)
    my_node = Node_Decision_Maker(node_decision_maker_name,
                                  topic_armor_pos,
                                  predict_freq,
                                  make_decision_freq,
                                  receive_from_electric_sys_freq)
    
    with Custome_Context(node_subscribe_name,my_node):
        rclpy.spin(my_node)
        
    rclpy.shutdown()
