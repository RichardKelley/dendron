from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class ForceSuccessNode(DecoratorNode):
    def __init__(self, name: str, child: TreeNode = None):
        super().__init__(name)
        self.name = name
        self.child_node = child

    def tick(self):
        self.child_node.execute_tick()
        return NodeStatus.SUCCESS