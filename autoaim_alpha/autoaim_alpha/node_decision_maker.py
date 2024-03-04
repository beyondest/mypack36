from . import *
from .os_op.basic import *
import rclpy
from rclpy.node import Node
from .decision_maker.ballistic_predictor import *
from .decision_maker.decision_maker import *

class Node_Decision_Maker(Node,Custom_Context_Obj):

    def __init__(self,
                 name):
        super().__init__(name)
        self.pub_ele_sys_com = self.create_publisher(topic_electric_sys_com['type'],
                                                topic_electric_sys_com['name'],
                                                topic_electric_sys_com['qos_profile'])
        
        self.sub_ele_sys_state = self.create_subscription(topic_electric_sys_state['type'],
                                                      topic_electric_sys_state['name'],
                                                      self.recv_from_ele_sys_callback,
                                                      topic_electric_sys_state['qos_profile'])
        
        self.sub_predict_pos = self.create_subscription(topic_armor_pos_corrected['type'],
                                                      topic_armor_pos_corrected['name'],
                                                      self.sub_predict_pos_callback,
                                                      topic_armor_pos_corrected['qos_profile'])
                                        
        self.ballestic = Ballistic_Predictor(mode,
                                             ballistic_predictor_config_yaml_path)
        
        self.decision_maker = Decision_Maker(mode,
                                             decision_maker_config_yaml_path)

        self.timer = self.create_timer(1/make_decision_freq, self.make_decision_callback)
        self.cur_yaw = 0
        self.cur_pitch = 0


    def recv_from_ele_sys_callback(self, msg:ElectricsysState):
        self.ballestic._update_camera_pos_in_gun_pivot_frame(msg.cur_yaw,msg.cur_pitch)
        try:
            zero_unix_offset = self.get_parameter("zero_unix_offset").get_parameter_value().double_value
        except:
            self.get_logger().error("Get zero_unix_offset parameter fail, use current time as zero_unix_offset")
            zero_unix_offset = time.time()
        
        minute,second,second_frac = TRANS_UNIX_TIME_TO_T(msg.unix_time,zero_unix_offset)
        self.decision_maker.update_our_side_info(msg.cur_yaw,
                                                  msg.cur_pitch,
                                                  minute,
                                                  second,
                                                  second_frac)

    def sub_predict_pos_callback(self, msg:ArmorPos):
        target_pos_in_camera_frame = np.array([msg.pose.pose.position.x,
                                                msg.pose.pose.position.y,
                                                msg.pose.pose.position.z])
        q = Quaternion(msg.pose.pose.orientation.w,
                       msg.pose.pose.orientation.x,
                       msg.pose.pose.orientation.y,
                       msg.pose.pose.orientation.z)
        rvec = q.get_axis() * q.angle
        self.decision_maker.update_enemy_side_info(msg.armor_name,
                                                   msg.armor_id,
                                                   target_pos_in_camera_frame,
                                                   rvec,
                                                   msg.confidence,
                                                   msg.pose.header.stamp.sec + msg.pose.header.stamp.nanosec/1e9
                                                   )
        self.get_logger().info(f"Update enemy side info {msg.armor_name}, id {msg.armor_id}")
        
    def make_decision_callback(self):
        target_armor = self.decision_maker.choose_target()
        
        rel_yaw,abs_pitch, flight_time, if_success = self.ballestic.get_fire_yaw_pitch(target_armor.tvec)
        if if_success:
            predict_time = target_armor.time
            
            com_msg = ElectricsysCom()
            com_msg.fire_times = 0
            com_msg.reach_unix_time = predict_time
            com_msg.target_abs_pitch = abs_pitch
            yaw = rel_yaw + self.cur_yaw
            if yaw > np.pi * 2:
                yaw -= np.pi * 2
            elif yaw < 0:
                yaw += np.pi * 2
            com_msg.target_abs_yaw = yaw
            com_msg.sof = 'A'
            com_msg.reserved_slot = 0
            self.pub_ele_sys_com.publish(com_msg)
            self.get_logger().info(f"Publish com msg: {com_msg}")
        else:
            self.get_logger().info(f"Get fire yaw pitch fail,not publish com msg, target pos: {target_armor.tvec}")

    
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"Node {self.get_name()} get error {exc_value}")
        
        

def main(args=None):
    rclpy.init(args=args)
    
    node = Node_Decision_Maker(node_decision_maker_name)
    
    with Custome_Context(node_decision_maker_name,node):
        rclpy.spin(node)
        
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
