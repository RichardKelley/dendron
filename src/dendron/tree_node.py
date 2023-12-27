from .blackboard import Blackboard
from .basic_types import NodeType, NodeStatus

import types

from typing import Dict, List, Callable
from dataclasses import dataclass

class TreeNode:
    def __init__(self, name : str):
        self.name = name
        self.blackboard = None
        self.status = NodeStatus.IDLE

        self.pre_tick_fn = None
        self.post_tick_fn = None

    def execute_tick(self) -> NodeStatus:
        if self.pre_tick_fn:
            self.pre_tick_fn()

        new_status = self.tick()

        if self.post_tick_fn:
            self.post_tick_fn()

        self.set_status(new_status)

        return new_status

    def set_description(self, desc):
        """
        A textual description intended to help with automated
        policy construction.
        """
        self.description = desc

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

    def set_pre_tick(self, f : Callable):
        self.pre_tick_fn = types.MethodType(f, self)

    def set_post_tick(self, f : Callable):
        self.post_tick_fn = types.MethodType(f, self)

    def tick(self):
        raise NotImplementedError("Tick should be implemented in a subclass.")

    def reset_status(self):
        self.status = NodeStatus.IDLE

    def pretty_repr(self, depth = 0):
        raise NotImplementedError("Pretty printing should be implemented in a subclass.")
