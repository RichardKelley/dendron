from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

import time

class Timeout(DecoratorNode):
    """
    The timeout decorator ticks its child and starts a timer. The next time
    it receives a tick it will check its timer against the time limit. If 
    the elapsed time exceeds the limit, this node returns `FAILURE`.

    This is primarily useful for nodes that are executing asynchronously.

    Args:
        name (`str`):
            The given name of this node.
        child (`dendron.tree_node.TreeNode`):
            The child of this node.
        timelimit (`int`):
            The integer number of *milliseconds* to wait before returning 
            failure.
    """
    def __init__(self, name: str, child: TreeNode, timelimit: int) -> None:
        super().__init__(name, child)
        self.timelimit = timelimit
        self.timer_started = False
        self.start_time = 0 # this is an int in millis.

    def reset(self) -> None:
        """
        Reset the timer and instruct the child node to reset.
        """
        self.timer_started = False
        self.start_time = 0
        self.reset_child()

    def tick(self) -> NodeStatus:
        """
        Tick the child node, but with a time limit. If the time limit
        is exceeded, return `FAILURE`.
        """
        if not self.timer_started:
            self.timer_started = True
            self.set_status(NodeStatus.RUNNING)
            self.start_time = time.time_ns()

        child_status = self.child_node.execute_tick()

        elapsed_ms = (time.time_ns() - self.start_time)/1e6
        if elapsed_ms > self.timelimit:
            return NodeStatus.FAILURE
        else:
            return child_status



