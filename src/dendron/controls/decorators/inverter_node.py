from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class InverterNode(DecoratorNode):
    def __init__(self, name):
        self.name = name

    def tick(self):
        pass