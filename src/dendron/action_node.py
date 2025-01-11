from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

from typing import Optional, List

import logging

class ActionNode(TreeNode):

    _used_names = set(["action"])

    """
    An action node encapsulates the notion of a self-contained action
    or behavior. The bulk of the observable actions of a behavior tree
    are due to the action nodes.

    `ActionNode`s are one of the two kinds of leaf nodes in a Behavior
    Tree - the other being the `ConditionNode`.

    Args:
        name (`str`):
            The given name of this node.
    """

    def __init__(self, name="action") -> None:
        super().__init__()
        self._name = None
        self.name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name is not None:
            ActionNode._used_names.remove(self._name)

        if value in ActionNode._used_names:
            suffix = 0
            new_name = f"{value}_{suffix}"
            while new_name in ActionNode._used_names:
                suffix += 1
                new_name = f"{value}_{suffix}"
            value = new_name
        
        ActionNode._used_names.add(value)
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
            `NodeType`: The type (`ACTION`).
        """
        return NodeType.ACTION

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
        repr = f"{tabs}Action {self.name}"
        return repr
    
