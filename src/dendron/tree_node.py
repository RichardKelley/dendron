from .blackboard import Blackboard
from .basic_types import NodeType, NodeStatus

import types

from typing import Dict, List, Callable, Optional, Self
from dataclasses import dataclass

import logging

class TreeNode:
    def __init__(self, name : str):
        self.name = name
        self.blackboard = None
        self.status = NodeStatus.IDLE

        self.pre_tick_fn = None
        self.post_tick_fn = None

        self.logger = None
        self.log_level = None

    def set_logger(self, new_logger):
        raise NotImplementedError("set_logger should be defined in subclass.")

    def set_log_level(self, new_level):
        raise NotImplementedError("set_log_level should be defined in subclass.")

    def _get_level_str(self, new_level):
        if self.logger is not None:
            level_str = "None"  
            match self.logger.level:    
                case logging.DEBUG: 
                    level_str = "debug" 
                case logging.INFO:  
                    level_str = "info"  
                case logging.WARNING:   
                    level_str = "warning"
                case logging.ERROR: 
                    level_str = "error" 
                case logging.CRITICAL:  
                    level_str = "critical"
            return level_str

    def execute_tick(self) -> NodeStatus:
        if self.logger is not None:
            log_fn = getattr(self.logger, self._get_level_str(self.log_level))
            log_fn(f"{self.name} - pre_tick")

        if self.pre_tick_fn is not None:
            self.pre_tick_fn()

        new_status = self.tick()

        if self.post_tick_fn is not None:
            self.post_tick_fn()

        self.set_status(new_status)

        if self.logger is not None:
            log_fn = getattr(self.logger, self._get_level_str(self.log_level))
            log_fn(f"{self.name} - post_tick {self.status}")

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

    def get_node_by_name(self, name : str) -> Optional[Self]:
        raise NotImplementedError("get_node_by_name should be implemented in a subclass.")

    def set_pre_tick(self, f : Callable):
        self.pre_tick_fn = types.MethodType(f, self)

    def set_post_tick(self, f : Callable):
        self.post_tick_fn = types.MethodType(f, self)

    def tick(self):
        raise NotImplementedError("Tick should be implemented in a subclass.")

    def reset(self):
        self.status = NodeStatus.IDLE

    def pretty_repr(self, depth = 0):
        raise NotImplementedError("Pretty printing should be implemented in a subclass.")
