from .blackboard import Blackboard
from .basic_types import NodeType, NodeStatus

import types

import typing
from typing import Dict, List, Callable, Optional, Self, Any
from dataclasses import dataclass

import logging

BehaviorTree = typing.NewType("BehaviorTree", None)

class TreeNode:
    """
    Base class for a node in a behavior tree.

    Args:
        name (`str`):
            The name to give to this node. 
    """
    
    def __init__(self, name : str) -> None:
        self.name = name
        self.blackboard = None
        self.status = NodeStatus.IDLE

        self.pre_tick_fn = None
        self.post_tick_fn = None

        self.logger = None
        self.log_level = None

        self.tree = None

    def set_tree(self, tree : BehaviorTree) -> None:
        """
        Set the tree that contains this node.

        Args:
            tree (`dendron.behavior_tree.BehaviorTree`)
        """
        self.tree = tree

    def set_logger(self, new_logger) -> None:
        raise NotImplementedError("set_logger should be defined in subclass.")

    def set_log_level(self, new_level) -> None:
        raise NotImplementedError("set_log_level should be defined in subclass.")

    def _get_level_str(self, new_level) -> str:
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
        """
        Performs pre-tick operations, calls the Node's tick() method, and 
        then performs post-tick operations. If logging is enabled, then this 
        is where log functions are called.

        Returns:
            `dendron.basic_types.NodeStatus`: The status returned by the inner 
            call to tick().
        """
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

    def set_description(self, desc) -> None:
        """
        A textual description intended to help with automated
        policy construction.

        Args:
            desc (`str`):
                The textual description of this node's functionality.
        """
        self.description = desc

    def halt_node(self) -> None:
        raise NotImplementedError("Halt behavior is specified in subclass.")

    def set_blackboard(self, bb : Blackboard) -> None:
        self.blackboard = bb

    # TODO consider deprecating
    def blackboard_set(self, key, value) -> None:
        full_key = self.name + '/' + key
        self.blackboard[full_key] = value

    # TODO consider deprecating
    def blackboard_get(self, key) -> Any:
        full_key = self.name + '/' + key
        return self.blackboard[full_key]

    def is_halted(self) -> bool:
        return self.status == NodeStatus.IDLE

    def get_status(self) -> NodeStatus:
        return self.status 

    def set_status(self, new_status) -> None:
        self.status = new_status

    def name(self) -> str:
        return self.name

    def node_type(self) -> NodeType:
        raise NotImplementedError("Type is specified in subclass.")

    def get_node_by_name(self, name : str) -> Optional[Self]:
        """
        Search for a node by its name.

        Args:
            name (`str`):
                The name of the node we are looking for.

        Returns:
            `Optional[TreeNode]`: Either a node with the given name,
            or None.
        """
        raise NotImplementedError("get_node_by_name should be implemented in a subclass.")

    def set_pre_tick(self, f : Callable) -> None:
        self.pre_tick_fn = types.MethodType(f, self)

    def set_post_tick(self, f : Callable) -> None:
        self.post_tick_fn = types.MethodType(f, self)

    def tick(self) -> NodeStatus:
        raise NotImplementedError("Tick should be implemented in a subclass.")

    def reset(self) -> None:
        self.status = NodeStatus.IDLE

    def pretty_repr(self, depth = 0) -> str:
        """
        Return a string representation of this node at the given depth.

        Args:
            depth (`int`):
                The depth of this node in a surrounding tree.

        Returns:
            `str`: The indented string representation.
        """
        raise NotImplementedError("Pretty printing should be implemented in a subclass.")
