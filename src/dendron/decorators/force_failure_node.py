from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class ForceFailureNode(DecoratorNode):
    def __init__(self, name: str, child: TreeNode = None) -> None:
        super().__init__(name)
        self.name = name
        self.child_node = child

    def tick(self) -> NodeStatus:
        self.child_node.execute_tick()
        return NodeStatus.FAILURE