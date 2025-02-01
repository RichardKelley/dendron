from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus
from dendron.registry import register_dendron_node

@register_dendron_node
class AlwaysFailure(ActionNode):
    """
    An action node that always returns `FAILURE`.

    Args:
        name (`str`):
            The given name of this node.
    """
    def __init__(self, name : str) -> None:
        super().__init__(name)

    def tick(self) -> NodeStatus:
        """
        Always return `FAILURE`.
        """
        return NodeStatus.FAILURE