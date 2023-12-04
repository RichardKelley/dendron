from ..action_node import ActionNode
from ..tree_node import NodeStatus

class SimpleActionNode(ActionNode):
    def __init__(self, name, callback):
        super().__init__(name)
        self.callback = callback

    def tick(self):
        return self.callback() 
