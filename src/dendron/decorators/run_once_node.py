from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class RunOnceNode(DecoratorNode):
    def __init__(self, name: str, child: TreeNode):
        super().__init__(name, child)
        self.has_run = False

    def reset_status(self):
        self.has_run = False
        self.status = NodeStatus.IDLE
        self.child_node.reset_status()

    def tick(self):
        self.set_status(NodeStatus.RUNNING)
        if self.has_run:
            return NodeStatus.SKIPPED

        self.has_run = True
        child_status = self.child_node.execute_tick()
        return child_status