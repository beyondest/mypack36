from . import *
from rclpy.node import Node
from .port_slavedevice.com_tools import *



class Node_Com(Node):
    def __init__(self,name:str):
        super().__init__(name)
        self.sub = self.create_subscription



