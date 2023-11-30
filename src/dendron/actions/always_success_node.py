from ..action_node import ActionNode
from ..tree_node import NodeStatus
from ..blackboard import Blackboard

class AlwaysSuccessNode(ActionNode):

    def __init__(self):
        super().__init__("AlwaysSuccess")

    def tick(self):
        return NodeStatus.SUCCESS
