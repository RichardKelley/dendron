from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class RunOnce(DecoratorNode):
    """
    The RunOnce decorator tracks whether or not its child has been
    ticked. If it has, the next time it is ticked the decorator will
    return a status of `SKIPPED`. It will continue to return that
    status until it is explicitly reset.

    Args:
        name (`str`):
            The given name of this node.
        child (`dendron.tree_node.TreeNode`):
            The child node.
    """
    def __init__(self, name: str, child: TreeNode) -> None:
        super().__init__(name, child)
        self.has_run = False

    def reset(self) -> None:
        """
        Reset the `has_run` tracker so that the child node can be
        `tick()`ed again.
        """
        self.has_run = False
        self.status = NodeStatus.IDLE
        self.child_node.reset_status()

    def tick(self) -> NodeStatus:
        """
        If the child node has been `tick()`ed already, return a status
        of `SKIPPED` without `tick()`ing the child again. Otherwise 
        `tick()` the child node and set `has_run` to `True` so that the
        node will not be `tick()`ed again.
        """
        self.set_status(NodeStatus.RUNNING)
        if self.has_run:
            return NodeStatus.SKIPPED

        self.has_run = True
        child_status = self.child_node.execute_tick()
        return child_status