from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

import time

class TimeoutNode(DecoratorNode):
    def __init__(self, name: str, child: TreeNode, timelimit: int):
        super().__init__(name, child)
        self.timelimit = timelimit
        self.timer_started = False
        self.start_time = 0 # this is an int in millis.

    def reset(self):
        self.timer_started = False
        self.start_time = 0
        self.reset_child()

    def tick(self):
        if not self.timer_startd:
            self.timer_started = True
            self.set_status(NodeStatus.RUNNING)
            self.start_time = time.time_ns()

        child_status = self.child_node.execute_tick()

        elapsed_ms = (time.time_ns() - self.start_time)/1e6
        if elapsed_ms > self.timelimit:
            return NodeStatus.FAILURE
        else:
            return child_status



