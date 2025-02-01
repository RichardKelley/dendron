from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus

from typing import Callable

from dendron.registry import register_dendron_node

@register_dendron_node
class SimpleAction(ActionNode):
    """
    A simple action node is initialized with a callback that is 
    called every time this node `tick()`s. The callback should
    be a function that that returns a `NodeStatus`.

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
