from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode
from .blackboard import Blackboard

import typing
from typing import List, Optional

import logging

BehaviorTree = typing.NewType("BehaviorTree", None)

class ControlNode(TreeNode):

    def __init__(self, name, children : List[TreeNode] = None) -> None:
        super().__init__(name)
        self.children : List[TreeNode] = children 

    def set_tree(self, tree : BehaviorTree) -> None:
        self.tree = tree
        for c in self.children:
            c.set_tree(tree)

    def set_logger(self, new_logger) -> None:
        self.logger = new_logger
        for c in self.children:
            c.set_logger(new_logger)

    def set_log_level(self, new_level) -> None:
        self.log_level = new_level
        for c in self.children:
            c.set_log_level(new_level)

    def add_child(self, child : TreeNode) -> None:
        self.children.append(child)

    def add_children(self, children : List[TreeNode]) -> None:
        self.children.extend(children)

    def set_blackboard(self, bb : Blackboard) -> None:
        self.blackboard = bb
        for child in self.children:
            child.set_blackboard(bb)

    def get_node_by_name(self, name : str) -> Optional[TreeNode]:
        if self.name == name:
            return self
        else:
            for child in self.children:
                node = child.get_node_by_name(name)
                if node != None:
                    return node
            return None

    def children_count(self) -> int:
        return len(self.children)

    def children(self) -> List[TreeNode]:
        return self.children

    def child(self, index : int) -> TreeNode:
        return self.children[index]

    def node_type(self) -> NodeType:
        return NodeType.CONTROL

    def halt_node(self) -> None:
        self.reset_children()
        self.reset_status()

    def reset(self) -> None:
        for child in self.children:
            child.reset()
    