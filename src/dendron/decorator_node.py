from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

import typing
from typing import Optional

import logging

BehaviorTree = typing.NewType("BehaviorTree", None)

class DecoratorNode(TreeNode):

    def __init__(self, name, child : TreeNode = None):
        super().__init__(name)
        self.child_node : TreeNode = child

    def set_logger(self, new_logger):
        self.logger = new_logger
        self.child_node.set_logger(new_logger)

    def set_log_level(self, new_level):
        self.log_level = new_level
        self.child_node.set_log_level(new_level)

    def node_type(self):
        return NodeType.DECORATOR 

    def set_child(self, child):
        self.child_node = child

    def get_child(self):
        return self.child_node

    def get_node_by_name(self, name : str) -> Optional[TreeNode]:
        if self.name == name:
            return self
        else:
            return self.child_node.get_node_by_name(name)

    def halt_child(self):
        self.child_node.halt_node()

    def set_tree(self, tree : BehaviorTree):
        self.tree = tree
        self.child_node.set_tree(tree)

    def reset(self):
        self.child_node.reset()

    def pretty_repr(self, depth = 0):
        tabs = '\t'*depth
        repr = f"{tabs}Decorator {self.name}\n{self.child_node.pretty_repr(depth+1)}"
        return repr