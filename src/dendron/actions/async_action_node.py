from ..action_node import ActionNode
from ..basic_types import NodeStatus

from typing import Callable

from concurrent import futures

class AsyncAction(ActionNode):
    """
    An action node that operates asynchronously. 

    Once ticked, the node enters a `RUNNING` state, which it
    remains in until the node's `Callable` returns with a status.
    That status is what gets returned by the tick function the next 
    time it is called.

    Internally, this node maintains a future to store the eventual
    result of the asynchronous computation.

    Asynchronous execution is handled by the node's tree's executor,
    which means this node cannot run without an enclosing tree.

    Args:
        name (`str`):
            The given name of this node.
        cb (`Callable`):
            The callable object that will be executed asynchronously.
    """

    def __init__(self, name : str, cb : Callable) -> None:
        super().__init__(name)

        self.cb = cb
        self.fut = None        

    def reset(self) -> None:
        """
        Set the status of this node to `IDLE`, and clear out the node's
        future.
        """
        self.status = NodeStatus.IDLE
        self.fut = None

    def tick(self) -> NodeStatus:
        """
        Asynchronously execute this node's callback.

        Returns:
            `NodeStatus`: The status contained in the node's future, or
            `RUNNING` if the node is not yet done.
        """
        if self.fut is None:
            self.fut = self.tree.executor.submit(self.cb)

        self.fut.add_done_callback(lambda f: self.set_status(f.result()))

        if self.fut.done():
            old_status = self.status
            self.reset()
            return old_status
        else:
            return NodeStatus.RUNNING
