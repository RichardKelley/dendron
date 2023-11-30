from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode
from .blackboard import Blackboard

from typing import List 

class ControlNode(TreeNode):

    def __init__(self, name):
        super().__init__(name)

        self.children : List[TreeNode] = []

    def add_child(self, child : TreeNode):
        self.children.append(child)

    def add_children(self, children : List[TreeNode]):
        self.children.extend(children)

    def set_blackboard(self, bb : Blackboard):
        self.blackboard = bb
        for child in self.children:
            child.set_blackboard(bb)

    def children_count(self) -> int:
        return len(self.children)

    def children(self) -> List[TreeNode]:
        return self.children

    def child(index : int) -> TreeNode:
        return self.children[index]

    def node_type(self) -> NodeType:
        return NodeType.CONTROL

    def halt_node(self):
        self.reset_children()
        self.reset_status()

    def reset_children(self):
        for child in self.children:
            if child.get_status() == NodeStatus.RUNNING:
                child.halt_node()
            child.reset_status()
    