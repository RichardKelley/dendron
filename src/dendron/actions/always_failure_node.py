from ..action_node import ActionNode
from ..tree_node import NodeStatus

class AlwaysFailureNode(ActionNode):
    def __init__(self, name) -> None:
        super().__init__(name)

    def tick(self) -> NodeStatus:
        return NodeStatus.FAILURE