from ..condition_node import ConditionNode
from ..tree_node import NodeStatus

class SimpleConditionNode(ConditionNode):
    def __init__(self, name, callback):
        super().__init__(name)
        self.callback = callback
    
    def tick(self):
        return self.callback()
