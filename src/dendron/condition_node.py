from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode

class ConditionNode(TreeNode):
    def __init__(self, name):
        super().__init__(name)

    def node_type(self) -> NodeType:
        return NodeType.CONDITION

    def pretty_repr(self, depth = 0):
        tabs = '\t'*depth
        repr = f"{tabs}Condition {self.name}"
        return repr
