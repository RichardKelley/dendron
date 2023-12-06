from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

class DecoratorNode(TreeNode):

    def __init__(self, name, child : TreeNode = None):
        super().__init__(name)

        self.child_node : TreeNode = child

    def node_type(self):
        return NodeType.DECORATOR 

    def set_child(self, child):
        self.child_node = child

    def get_child(self):
        return self.child_node

    def halt_child(self):
        self.child_node.halt_node()

    def reset_child(self):
        self.child_node.reset_status()

    def pretty_repr(self, depth = 0):
        tabs = '\t'*depth
        repr = f"{tabs}Decorator {self.name}\n{self.child_node.pretty_repr(depth+1)}"
        return repr