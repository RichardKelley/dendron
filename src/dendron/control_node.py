from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode
from .blackboard import Blackboard

import typing
from typing import List, Optional

import logging

BehaviorTree = typing.NewType("BehaviorTree", None)

class ControlNode(TreeNode):
    """
    Base class for a control node.

    A control node maintains a list of children that it ticks under
    some conditions. The node tracks the state of its children as 
    they tick, and decides whether or not to continue based on its 
    internal logic.

    Args:
        name (`str`):
            The given name of this control node.
        children (`List[TreeNode]`):
            An optional initial list of children.
    """

    def __init__(self, name : str, children : List[TreeNode] = None) -> None:
        super().__init__(name)
        self.children : List[TreeNode] = children 

    def set_tree(self, tree : BehaviorTree) -> None:
        """
        Set the tree of this node, and then have each of the children
        set their tree similarly.

        Args:
            tree (`dendron.behavior_tree.BehaviorTree`):
                The tree that will contain this node.
        """
        self.tree = tree
        for c in self.children:
            c.set_tree(tree)

    def set_logger(self, new_logger) -> None:
        """
        Set the logger for this node, and then forward the logger to the
        children.

        Args:
            new_logger (`logging.Logger`):
                The Logger to use.
        """
        self.logger = new_logger
        for c in self.children:
            c.set_logger(new_logger)

    def set_log_level(self, new_level) -> None:
        """
        Set the log level for this node, then forward that level for the
        children to use.
        """
        self.log_level = new_level
        for c in self.children:
            c.set_log_level(new_level)

    def add_child(self, child : TreeNode) -> None:
        """
        Add a new child node to the end of the list.

        Args:
            child (`dendron.tree_node.TreeNode`):
                The new child node.
        """
        self.children.append(child)

    def add_children(self, children : List[TreeNode]) -> None:
        """
        Add a list of children to the end of the list.

        Args:
            children (`List[TreeNode]`):
                The list of `TreeNode`s to add. 
        """
        self.children.extend(children)

    def set_blackboard(self, bb : Blackboard) -> None:
        """
        Set the blackboard for this node, and then forward to the
        children.

        Args:
            bb (`dendron.blackboard.Blackboard`):
                The new blackboard to use.
        """
        self.blackboard = bb
        for child in self.children:
            child.set_blackboard(bb)

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
            for child in self.children:
                node = child.get_node_by_name(name)
                if node != None:
                    return node
            return None

    def children_count(self) -> int:
        """
        Get the current number of children.

        Returns:
            `int`: The length of the children list.
        """
        return len(self.children)

    def children(self) -> List[TreeNode]:
        """
        Get the list of children.

        Returns:
            `List[TreeNode]`: The `self.children` list.
        """
        return self.children

    def child(self, index : int) -> TreeNode:
        """
        Get the child that is at position `index` in the list. Does
        not perform bounds checking.

        Args:
            index (`int`):
                The index of the child we want.

        Returns:
            `TreeNode`: The child at the desired index.
        """
        return self.children[index]

    def node_type(self) -> NodeType:
        """
        Return this node's `NodeType`.

        Returns:
            `NodeType`: The type (`CONTROL`).
        """
        return NodeType.CONTROL

    def halt_node(self) -> None:
        """
        Reset the children and then reset this node.
        """
        self.reset_children()
        self.reset_status()

    def reset(self) -> None:
        """
        Instruct each child to reset.
        """
        for child in self.children:
            child.reset()
    