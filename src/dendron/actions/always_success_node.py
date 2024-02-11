from ..action_node import ActionNode
from ..basic_types import NodeStatus
from ..blackboard import Blackboard

class AlwaysSuccess(ActionNode):
    """
    An action node that always returns `SUCCESS`.

    Args:
        name (`str`):
            The given name of this node.
    """
    def __init__(self, name : str) -> None:
        super().__init__(name)

    def tick(self) -> NodeStatus:
        """
        Always return `SUCCESS`.
        """
        return NodeStatus.SUCCESS
