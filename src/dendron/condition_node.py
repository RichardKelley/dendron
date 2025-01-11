from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

from typing import Optional, List

import logging

class ConditionNode(TreeNode):

    _used_names = set(["condition"])

    """
    A condition node is a node that always *must* return either `SUCCESS` 
    or `FAILURE` - it can never be left in a `RUNNING` state. Such nodes
    are intended to model boolean conditions (hence the name). 

    `ConditionNode`s are one of the two kinds of leaf nodes in a Behavior
    Tree - the other being the `ActionNode`.

    Args:
        name (`str`):
            The given name of this node.
    """
    
    def __init__(self, name : str = "condition") -> None:
        super().__init__()
        self._name = None
        self.name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name is not None:
            ConditionNode._used_names.remove(self._name)

        if value in ConditionNode._used_names:
            suffix = 0
            new_name = f"{value}_{suffix}"
            while new_name in ConditionNode._used_names:
                suffix += 1
                new_name = f"{value}_{suffix}"
            value = new_name
        
        ConditionNode._used_names.add(value)
        self._name = value

    def children(self) -> List[TreeNode]:
        return []

    def set_logger(self, new_logger) -> None:
        """
        Set the logger for this node.
        """
        self.logger = new_logger

    def set_log_level(self, new_level) -> None:
        """
        Set the log level for this node.
        """
        self.log_level = new_level

    def node_type(self) -> NodeType:
        """
        Get the type of this node.

        Returns:
            `NodeType`: The type (`CONDITION`).
        """
        return NodeType.CONDITION

    def get_node_by_name(self, name : str) -> Optional[TreeNode]:
        """
        Search for a node by its name.

        Args:
            name (`str`):
                The name of the node we are looking for.

        Returns:
            `Optional[TreeNode]`: Either a node with the given name,
            or None.
        """
        if self.name == name:
            return self
        else:
            return None

    def pretty_repr(self, depth = 0) -> str:
        """
        Return a string representation of this node at the given depth.

        Args:
            depth (`int`):
                The depth of this node in a surrounding tree.

        Returns:
            `str`: The indented string representation.
        """
        tabs = '\t'*depth
        repr = f"{tabs}Condition {self.name}"
        return repr
