from ..action_node import ActionNode
from ..basic_types import NodeStatus
from ..blackboard import Blackboard

class AlwaysSuccessNode(ActionNode):

    def __init__(self, name) -> None:
        super().__init__(name)

    def tick(self) -> NodeStatus:
        return NodeStatus.SUCCESS
