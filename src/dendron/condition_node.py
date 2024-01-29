from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

from typing import Optional

import logging

class ConditionNode(TreeNode):
    def __init__(self, name) -> None:
        super().__init__(name)

    def set_logger(self, new_logger) -> None:
        self.logger = new_logger

    def set_log_level(self, new_level) -> None:
        self.log_level = new_level

    def node_type(self) -> NodeType:
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
