from ..action_node import ActionNode
from ..tree_node import NodeStatus
from ..blackboard import Blackboard

class AlwaysSuccessNode(ActionNode):

    def __init__(self, name):
        super().__init__(name)

    def tick(self):
        return NodeStatus.SUCCESS
