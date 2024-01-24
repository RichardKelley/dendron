from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class RetryNode(DecoratorNode):
    def __init__(self, name: str, child: TreeNode, n_times: int) -> None:
        super().__init__(self, child)
        self.n_times = n_times
        self.retry_ct = 0

    def reset(self) -> None:
        self.retry_ct = 0
        self.child_node.reset()

    def tick(self) -> NodeStatus:
        should_retry = True
        
        while should_retry:
            child_status = self.child_node.execute_tick()
            match child_status:
                case NodeStatus.SUCCESS:
                    self.retry_ct = 0
                    self.reset_child()
                    return NodeStatus.SUCCESS
                case NodeStatus.FAILURE:
                    self.retry_ct += 1
                    should_retry = self.retry_ct < self.n_times
                case NodeStatus.RUNNING:
                    return NodeStatus.RUNNING

        self.retry_ct = 0
        return NodeStatus.FAILURE
