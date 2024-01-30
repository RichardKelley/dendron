from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

import typing
from typing import Optional

import logging

BehaviorTree = typing.NewType("BehaviorTree", None)

class DecoratorNode(TreeNode):
    """
    A decorator is a "wrapper" around a single node. The purpose of 
    the decorator is to modify or support the action of its child in
    some way.

    Args:
        name (`str`):
            The given name of this node.
        child (`dendron.tree_node.TreeNode`):
            An optional child node. If not specified, the child must be 
            set before this node's `tick()` function is first called.
    """

    def __init__(self, name, child : TreeNode = None) -> None:
        super().__init__(name)
        self.child_node : TreeNode = child

    def set_logger(self, new_logger) -> None:
        """
        Set the logger for this node, and then forward the logger to the 
        child node.
        """
        self.logger = new_logger
        self.child_node.set_logger(new_logger)

    def set_log_level(self, new_level) -> None:
        """
        Set the log level for this node, and then forward that level to 
        the child node.
        """
        self.log_level = new_level
        self.child_node.set_log_level(new_level)

    def node_type(self) -> NodeType:
        """
        Return this node's type.
        """
        return NodeType.DECORATOR 

    def set_child(self, child : TreeNode) -> None:
        """
        Set the child of this node to a new `TreeNode`.

        Args:
            child (`dendron.tree_node.TreeNode`):
                The new child of this decorator.
        """
        self.child_node = child

    def get_child(self) -> TreeNode:
        """
        Get the child of this decorator.

        Returns:
            `TreeNode`: The child of this node.
        """
        return self.child_node

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
            return self.child_node.get_node_by_name(name)

    def halt_child(self) -> None:
        """
        Instruct the child node to halt.
        """
        self.child_node.halt_node()

    def set_tree(self, tree : BehaviorTree) -> None:
        """
        Set the tree of this node, and then forward the tree to the child
        to have it set its tree.

        Args:
            tree (`dendron.behavior_tree.BehaviorTree`):
                The tree that contains this node.
        """
        self.tree = tree
        self.child_node.set_tree(tree)

    def reset(self) -> None:
        """
        Set the status of this node to IDLE and instruct the child node to
        reset.
        """
        self.node_status = NodeStatus.IDLE
        self.child_node.reset()

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
        repr = f"{tabs}Decorator {self.name}\n{self.child_node.pretty_repr(depth+1)}"
        return repr