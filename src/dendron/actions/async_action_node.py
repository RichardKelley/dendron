from ..action_node import ActionNode
from ..basic_types import NodeStatus

from typing import Callable

from concurrent import futures

class AsyncActionNode(ActionNode):
    def __init__(self, name : str, cb : Callable, num_workers : int = 4):
        super().__init__(name)

        self.cb = cb
        self.fut = None        

    def reset(self):
        self.status = NodeStatus.IDLE
        self.fut = None

    def tick(self):
        if self.fut is None:
            self.fut = self.tree.executor.submit(self.cb)

        self.fut.add_done_callback(lambda f: self.set_status(f.result()))

        if self.fut.done():
            old_status = self.status
            self.reset()
            return old_status
        else:
            return NodeStatus.RUNNING
