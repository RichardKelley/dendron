from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..control_node import ControlNode

class FallbackNode(ControlNode):

    def __init__(self, name):
        super().__init__(name)

        self.current_child_idx = 0

    def halt_node(self):
        self.current_child_idx = 0
        ControlNode.halt(self)

    def tick(self):
        pass