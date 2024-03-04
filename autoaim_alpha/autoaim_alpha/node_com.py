from . import *
from .os_op.basic import *
import rclpy
from rclpy.node import Node
from .port_slavedevice.port import *
import time
class Node_Com(Node,Custom_Context_Obj):

    def __init__(self,
                 name
                 ):
        
        super().__init__(name)
        self.publisher_ = self.create_publisher(topic_electric_sys_state['type'],
                                                topic_electric_sys_state['name'],
                                                topic_electric_sys_state['qos_profile'])
        
        self.subscription_ = self.create_subscription(topic_electric_sys_com['type'],
                                                      topic_electric_sys_com['name'],
                                                      self.listener_callback,
                                                      topic_electric_sys_com['qos_profile'])
                                              
        self.timer_send_msg = self.create_timer(1/send_msg_freq, self.timer_send_msg_callback)
        self.timer_recv_msg = self.create_timer(1/recv_from_ele_sys_freq, self.timer_recv_msg_callback)
        self.port = Port(mode,
                         port_config_yaml_path)
        
        self.init_synchronization_time()
        self.last_sub_topic_time = 0
        
        

    def listener_callback(self, msg: ElectricsysCom):
        
        if msg.sof == 'A' :
            self.last_sub_topic_time = time.time()
            self.port.action_data.fire_times = msg.fire_times
            self.port.action_data.abs_pitch_10000 = int(msg.target_abs_pitch * 10000)
            self.port.action_data.abs_yaw_10000 = int((msg.target_abs_yaw - np.pi) * 10000)  # due to int16 is from -32768 to 32767, so we need to convert angle to this range
            self.port.action_data.reserved_slot = msg.reserved_slot
            minute, second, second_frac = TRANS_UNIX_TIME_TO_T(msg.reach_unix_time, self.zero_unix_time)
            self.port.action_data.target_minute = minute
            self.port.action_data.target_second = second
            self.port.action_data.target_second_frac_10000 = int(second_frac * 10000)
            self.get_logger().info(f"SOF A from Decision maker")
            
            if msg.fire_times > 0:
                self.port.send_msg(msg.sof)
                self.port.action_data.fire_times = 0
                self.get_logger().info(f"Fire : {msg.fire_times} times")
        
                
        elif msg.sof == 'S':
            cur_unix_time = time.time()
            minute, second, second_frac = TRANS_UNIX_TIME_TO_T(cur_unix_time, self.zero_unix_time)
            self.port.syn_data.present_minute = minute
            self.port.syn_data.present_second = second
            self.port.syn_data.present_second_frac_10000 = int(second_frac * 10000)
            self.port.send_msg(msg.sof)
            self.get_logger().debug(f"Sync time : {cur_unix_time:.3f}")
            self.get_logger().info(f"SOF S from Decision maker")
            
            
        else:
            self.get_logger().error(f"Unknown sof {msg.sof}")
        
    def init_synchronization_time(self):
        self.port.syn_data.present_minute = 0
        self.port.syn_data.present_second = 0
        self.port.syn_data.present_second_frac_10000  = 0
        self.zero_unix_time = time.time()
        
        self.declare_parameter('zero_unix_time',self.zero_unix_time)
        self.port.send_msg('S')
        self.get_logger().info(f"Init synchronization time")
        

        
    def timer_send_msg_callback(self):
        cur_time = time.time()
        if cur_time - self.last_sub_topic_time > 0.5:
            next_time = cur_time + self.port.params.communication_delay
            self.port.action_data.fire_times = 0
            self.port.action_data.abs_pitch_10000 = int(self.port.pos_data.present_pitch * 10000)
            self.port.action_data.abs_yaw_10000 = int((self.port.pos_data.present_yaw - np.pi) * 10000)  # due to int16 is from -32768 to 32767, so we need to convert angle to this range
            self.port.action_data.reserved_slot = 0
            minute, second, second_frac = TRANS_UNIX_TIME_TO_T(next_time, self.zero_unix_time)
            self.port.action_data.target_minute = minute
            self.port.action_data.target_second = second
            self.port.action_data.target_second_frac_10000 = int(second_frac * 10000)
            self.port.send_msg('A')
            self.get_logger().debug(f"Decision too old, last sub time {self.last_sub_topic_time:.3f}, cur_time {cur_time:.3f}")
        else:
            self.port.send_msg('A')
            self.get_logger().debug(f"Send action message")
    
    def timer_recv_msg_callback(self):
        if_error, current_yaw, current_pitch, cur_time_minute, cur_time_second, cur_time_second_frac = self.port.recv_feedback()
        if if_error:
            self.get_logger().error(f"Com Error ")
        else:
            msg = ElectricsysState()
            msg.cur_pitch = current_pitch
            msg.cur_yaw = current_yaw
            unix_time = TRANS_T_TO_UNIX_TIME(cur_time_minute, cur_time_second, cur_time_second_frac, self.zero_unix_time)
            msg.unix_time = unix_time
            self.publisher_.publish(msg)
            self.get_logger().info(f"Electric sys state p: {msg.cur_pitch:.3f}, y: {msg.cur_yaw:.3f}, t:{msg.unix_time:.3f}")
            
        
    def _start(self):
        try:
            self.port.port_open()
        except Exception as e:
            self.get_logger().error(f"Port open error {e}")
        self.get_logger().info(f"Node {self.get_name()} start success")
    
    def _end(self):
        try:
            self.port.port_close()
        except Exception as e:
            self.get_logger().error(f"Port close error {e}")
        self.get_logger().info(f"Node {self.get_name()} end success")
        self.destroy_node()

    def _errorhandler(self,exc_value):
        print(f"Node {self.get_name()} get error {exc_value}")
        
    

def main(args=None):
    rclpy.init(args=args)
    
    node = Node_Com(node_com_name)
    with Custome_Context(node_com_name,node):
        rclpy.spin(node=node)
        
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
