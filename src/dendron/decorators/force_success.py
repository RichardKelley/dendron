from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class ForceSuccess(DecoratorNode):
    """
    A Force Success node calls the `tick()` method of its child,
    ignores the result, and always returns a status of `SUCCESS`.

    May be useful for debugging.

    Args:
        name (`str`):
            The given name of this node.
        child (`dendron.tree_node.TreeNode`):
            An optional child node. If `None`, it is the responsibility
            of the caller to ensure that the `child_node` member 
            variable is set prior to the first `tick()` call.
    """
    def __init__(self, name: str, child: TreeNode = None) -> None:
        super().__init__(name, child)

    def tick(self) -> NodeStatus:
        """
        Instruct the child node to execute its `tick()` method, ignore
        the result, and return `SUCCESS`.
        """
        self.child_node.execute_tick()
        return NodeStatus.SUCCESS