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
                                        
        self.ballestic = Ballistic_Predictor(node_decision_maker_mode,
                                             ballistic_predictor_config_yaml_path)
        
        self.decision_maker = Decision_Maker(node_decision_maker_mode,
                                             decision_maker_config_yaml_path,
                                             enemy_car_list)

        self.timer = self.create_timer(1/make_decision_freq, self.make_decision_callback)
        self.cur_yaw = 0
        self.cur_pitch = 0
        if node_decision_maker_mode == 'Dbg':
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)


    def recv_from_ele_sys_callback(self, msg:ElectricsysState):
        
        self.ballestic._update_camera_pos_in_gun_pivot_frame(msg.cur_yaw,msg.cur_pitch)
        zero_unix_offset = self.get_parameter("zero_unix_offset").get_parameter_value().double_value

        
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
        

        
    def make_decision_callback(self):
        target_armor = self.decision_maker.choose_target()
        
        rel_yaw,abs_pitch, flight_time, if_success = self.ballestic.get_fire_yaw_pitch(target_armor.tvec,
                                                                                       self.cur_yaw,
                                                                                       self.cur_pitch)
        
        if if_success:
            predict_time = target_armor.time
            
            com_msg = ElectricsysCom()
            com_msg.fire_times = 0
            com_msg.reach_unix_time = predict_time
            com_msg.target_abs_pitch = abs_pitch
            yaw = rel_yaw + self.cur_yaw
            if yaw > np.pi:
                yaw -= np.pi * 2
            elif yaw < -np.pi:
                yaw += np.pi * 2
            com_msg.target_abs_yaw = yaw
            com_msg.sof = 'A'
            com_msg.reserved_slot = 0
            self.pub_ele_sys_com.publish(com_msg)
            
            
            if node_decision_maker_mode == 'Dbg':
                self.get_logger().debug(f"Choose Target {target_armor.name} id {target_armor.id} tvec {target_armor.tvec} rvec {target_armor.rvec} time {target_armor.time} ")
                self.get_logger().debug(f"Make decision : fire_times {com_msg.fire_times}  target_abs_pitch {com_msg.target_abs_pitch:.3f} target_abs_yaw {com_msg.target_abs_yaw:.3f} reach_unix_time {com_msg.reach_unix_time:.3f}")
                    
        else:
            self.get_logger().warn(f"Get fire yaw pitch fail,not publish com msg, target pos: {target_armor.tvec}")

    
    def _start(self):
        
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):

        self.get_logger().error(f"Node {self.get_name()} get error {exc_value}")
        
        

def main(args=None):
    rclpy.init(args=args)
    
    node = Node_Decision_Maker(node_decision_maker_name)
    
    with Custome_Context(node_decision_maker_name,node,[KeyboardInterrupt]):
        rclpy.spin(node)
        
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
