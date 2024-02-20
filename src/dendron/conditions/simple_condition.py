from ..condition_node import ConditionNode
from ..tree_node import NodeStatus

from typing import Callable

class SimpleCondition(ConditionNode):
    """
    A simple condition node is initialized with a callback that is
    called every time this node `tick()`s. The callback should be
    a function that returns a `NodeStatus`. Additionally, as a 
    condition node the callback should never return a status of 
    `RUNNING`. It is up to the caller to ensure that this invariant
    holds.

    Args:
        name (`str`):
            The given name of this node.
        callback (`Callable`):
            The callback to be executed upon every `tick()`.
    """
    def __init__(self, name : str, callback : Callable) -> None:
        super().__init__(name)
        self.callback = callback
    
    def tick(self) -> NodeStatus:
        """
        Call the callback function and return its status as the 
        node status.
        """
        return self.callback()
