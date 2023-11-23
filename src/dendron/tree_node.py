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

    def executeTick(self) -> NodeStatus:
        pass # TODO

    def haltNode(self):
        pass # TODO

    def isHalted(self) -> bool:
        return self.status == NodeStatus.IDLE

    def name(self) -> str:
        return self.name

    def config(self) -> NodeConfig:
        return self.config

    def node_type(self) -> NodeType:
        raise NotImplementedError()
    
    def tick(self):
        raise NotImplementedError()
