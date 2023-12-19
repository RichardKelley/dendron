from .blackboard import Blackboard
from .basic_types import NodeType, NodeStatus

from typing import Dict, List
from dataclasses import dataclass

class TreeNode:
    def __init__(self, name : str):
        self.name = name
        self.blackboard = None
        self.input_ports = []
        self.output_ports = []
        self.status = NodeStatus.IDLE
        self.registration_id = ""

    def execute_tick(self) -> NodeStatus:
        # TODO check preconditions if any 
        
        new_status = self.tick()

        # TODO check postconditions if any

        self.set_status(new_status)

        return new_status

    def halt_node(self):
        raise NotImplementedError("Halt behavior is specified in subclass.")

    def set_blackboard(self, bb : Blackboard):
        self.blackboard = bb

    def blackboard_set(self, key, value):
        full_key = self.name + '/' + key
        self.blackboard[full_key] = value

    def blackboard_get(self, key):
        full_key = self.name + '/' + key
        return self.blackboard[full_key]

    def add_input_port(self, input_port):
        self.input_ports.append(input_port)

    def add_output_port(self, output_port):
        self.output_ports.append(output_port)

    def is_halted(self) -> bool:
        return self.status == NodeStatus.IDLE

    def get_status(self) -> NodeStatus:
        return self.status 

    def set_status(self, new_status):
        self.status = new_status

    def name(self) -> str:
        return self.name

    def node_type(self) -> NodeType:
        raise NotImplementedError("Type is specified in subclass.")
    
    def get_input(self, key : str):
        if self.blackboard is not None:
            value = self.blackboard[key]
            return value
        else:
            raise RuntimeError("Blackboard is not set.")

    def set_output(self, key : str, value):
        if self.blackboard is not None:
            self.blackboard[key] = value
        else:
            raise RuntimeError("Blackboard is not set.")

    def tick(self):
        raise NotImplementedError("Tick should be implemented in a subclass.")

    def reset_status(self):
        self.status = NodeStatus.IDLE

    def pretty_repr(self, depth = 0):
        raise NotImplementedError("Pretty printing should be implemented in a subclass.")
