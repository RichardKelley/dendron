from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

class ActionNode(TreeNode):
    def __init__(self, name):
        super().__init__(name)

    def node_type(self) -> NodeType:
        return NodeType.ACTION
    
