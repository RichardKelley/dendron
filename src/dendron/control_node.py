from .basic_types import NodeType, NodeStatus
from .tree_node import NodeConfig, TreeNode

from typing import List 

class ControlNode(TreeNode):

    def __init__(self, name, cfg):
        super().__init__(name, cfg)

        self.children : List[TreeNode] = []

    def add_child(self, child : TreeNode):
        self.children.append(child)

    def children_count(self) -> int:
        return len(self.children)

    def children(self) -> List[TreeNode]:
        return self.children

    def child(index : int) -> TreeNode:
        return self.children[index]

    def node_type(self) -> NodeType:
        return NodeType.CONTROL

    def halt(self):
        self.reset_children()
        self.reset_status()

    def reset_children(self):
        for child in self.children:
            if child.get_status() == NodeStatus.RUNNING:
                child.halt_node()
            child.reset_status()
    