from ..basic_types import NodeType, NodeStatus
from ..tree_node import NodeConfig, TreeNode

class SequenceNode(ControlNode):

    def __init__(self, name):
        super().__init__(name, {})

        self.current_child_idx = 0

    def halt(self):
        self.current_child_idx = 0
        ControlNode.halt(self)

    def tick(self):
        pass # TODO

