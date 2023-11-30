from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

class DecoratorNode(TreeNode):

    def __init__(self, name):
        super().__init__(name)

        self.child_node : TreeNode = Node 

    def set_child(self, child):
        self.child_node = child

    def get_child(self):
        return self.child_node

    def halt_child(self):
        self.child.halt_node()

    def reset_child(self):
        self.child.reset_status()