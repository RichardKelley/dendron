from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..condition_node import ConditionNode

class GoalNode(ConditionNode):
    """
    EXPERIMENTAL! DO NOT USE!

    A GoalNode is just syntactic sugar for a condition node -

    It should return SUCCESS precisely when the goal is achieved.

    This may be useful for display and debugging purposes.
    """
    def __init__(self, name):
        super().__init__(name)

    def node_type(self) -> NodeType:
        return NodeType.GOAL 