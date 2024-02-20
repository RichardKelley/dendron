from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class ForceFailure(DecoratorNode):
    """
    A Force Failure decorator will tick its child, and regardless of the
    result will return failure.

    May be useful for debugging.

    Args:
        name (`str`):
            The given name of this node.
        child (`dendron.tree_node.TreeNode`):
            An optional child node. If `None`, the caller is responsible for
            setting the `child_node` member variable before `tick()` is
            called.
    """
    def __init__(self, name: str, child: TreeNode = None) -> None:
        super().__init__(name)
        self.name = name
        self.child_node = child

    def tick(self) -> NodeStatus:
        """
        Call the `tick()` method of the child and then return a status of
        `FAILURE`.
        """
        self.child_node.execute_tick()
        return NodeStatus.FAILURE