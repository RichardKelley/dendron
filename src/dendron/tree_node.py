from .blackboard import Blackboard
from .basic_types import NodeType, NodeStatus

from typing import Dict
from dataclasses import dataclass

PortsRemapping = Dict[str, str]

@dataclass
class NodeConfig:
    blackboard : Blackboard
    input_ports : PortsRemapping
    output_ports : PortsRemapping
    uid : int
    path : str
    
class TreeNode:
    def __init__(self, name : str, cfg : NodeConfig):
        self.name = name
        self.config = cfg
        self.status = NodeStatus.IDLE
        self.registration_id = ""

    def execute_tick(self) -> NodeStatus:
        # TODO check preconditions if any 
        
        new_status = self.tick()

        # TODO check postconditions if any

        self.set_status(new_status)

        return new_status

    def halt_node(self):
        pass # TODO

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

    def get_uid(self):
        return self.config.uid

    def full_path(self):
        return self.config.path

    def config(self) -> NodeConfig:
        return self.config
    
    def get_input(self, key : str):
        pass # TODO

    def set_output(self, key : str, value):
        pass # TODO

    def tick(self):
        raise NotImplementedError("Tick should be implemented in a subclass.")

    def reset_status(self):
        self.status = NodeStatus.IDLE
