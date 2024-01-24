from ..action_node import ActionNode
from ..basic_types import NodeStatus

class SimpleActionNode(ActionNode):
    def __init__(self, name, callback) -> None:
        super().__init__(name)
        self.callback = callback

    def tick(self) -> NodeStatus:
        return self.callback() 
