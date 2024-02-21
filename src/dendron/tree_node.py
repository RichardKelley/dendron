from .blackboard import Blackboard
from .basic_types import NodeType, NodeStatus

import types

import typing

# see the note below on why we don't want to use Self.
#from typing import Dict, List, Callable, Optional, Self, Any

from typing import Dict, List, Callable, Optional, Any
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

        self.pre_tick_fns = []
        self.post_tick_fns = []

        self.logger = None
        self.log_level = None

        self.tree = None

    def set_tree(self, tree : BehaviorTree) -> None:
        """
        Set the tree that contains this node.

        Args:
            tree (`dendron.behavior_tree.BehaviorTree`):
                The new tree this node is a part of.
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

        for f in self.pre_tick_fns:
            f()

        self.status = self.tick()

        for f in self.post_tick_fns:
            f()

        if self.logger is not None:
            log_fn = getattr(self.logger, self._get_level_str(self.log_level))
            log_fn(f"{self.name} - post_tick {self.status}")

        return self.status

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
        """
        Set the blackboard to be used by this TreeNode.

        Args:
            bb (`dendron.blackboard.Blackboard`):
                The new blackboard.
        """
        self.blackboard = bb

    def is_halted(self) -> bool:
        """
        Query whether this node is in a halted state.

        Returns:
            `bool`: True iff the status is IDLE.
        """
        return self.status == NodeStatus.IDLE

    def get_status(self) -> NodeStatus:
        """
        Get the current status of this node.

        Returns:
            `dendron.basic_types.NodeStatus`: The node status.
        """
        return self.status 

    def set_status(self, new_status : NodeStatus) -> None:
        """
        Set the node status to a new value.

        Args:
            new_status (`dendron.basic_types.NodeStatus`):
                The new NodeStatus.
        """
        self.status = new_status

    def name(self) -> str:
        """
        Get this node's human-readable name.

        Returns:
            `str`: The given name of this node.
        """
        return self.name

    def node_type(self) -> NodeType:
        raise NotImplementedError("Type is specified in subclass.")

    # the problem with this is that Self is only supported in 3.11+, which 
    # doesn't work with some libraries and packages.
    #def get_node_by_name(self, name : str) -> Optional[Self]:
    
    def get_node_by_name(self, name: str):
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

    def add_pre_tick(self, f : Callable) -> None:
        """
        Specify a function-like object to be called before the `tick()` 
        function. The argument is added to a list of such functions.        

        Args:
            f (`Callable`): 
                The function to call before `tick()`.
        """
        self.pre_tick_fns.append(types.MethodType(f, self))

    def add_post_tick(self, f : Callable) -> None:
        """
        Specify a function-like object to be called after the `tick()`
        function. The argument is added to a list of such functions.

        Args:
            f (`Callable`):
                The function to call after `tick()`.
        """
        self.post_tick_fns.append(types.MethodType(f, self))

    def tick(self) -> NodeStatus:
        raise NotImplementedError("Tick should be implemented in a subclass.")

    def reset(self) -> None:
        """
        Set the status of this node to IDLE.
        """
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
