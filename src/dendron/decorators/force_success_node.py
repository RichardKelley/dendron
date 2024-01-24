from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class ForceSuccessNode(DecoratorNode):
    def __init__(self, name: str, child: TreeNode = None) -> None:
        super().__init__(name, child)

    def tick(self) -> NodeStatus:
        self.child_node.execute_tick()
        return NodeStatus.SUCCESS