from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class Repeat(DecoratorNode):
    """
    A Repeat node ticks its child node repeatedly as long as the child
    continues to return `SUCCESS`. 

    Args:
        name (`str`):
            The given name of this node.
        child (`dendron.tree_node.TreeNode`):
            Optional child node. If `None`, it is the responsibility of the
            caller to ensure that the `child_node` member variable is set
            before the first `tick()` call.
        n_times (`int`):
            Number of times to `tick()` the child node if it continues
            to return `SUCCESS`.
    """
    def __init__(self, name : str, child : TreeNode, n_times : int) -> None:
        super().__init__(name, child)
        self.n_times = n_times
        self.repeat_ct = 0

    def reset(self) -> None:
        """
        Set the repeat counter to 0 and instruct the child node to reset.
        """
        self.repeat_ct = 0
        self.child_node.reset()

    def tick(self) -> NodeStatus:
        """
        Tick the child node until either it returns `FAILURE` or the child
        is `tick()`ed `n_times`.
        """
        should_repeat = True

        while should_repeat:
            child_status = self.child_node.execute_tick()
            match child_status:
                case NodeStatus.SUCCESS:
                    self.repeat_ct += 1
                    should_repeat = self.repeat_ct < self.n_times
                case NodeStatus.FAILURE:
                    self.repeat_ct = 0
                    self.reset_child()
                    return NodeStatus.FAILURE
                case NodeStatus.RUNNING:
                    return NodeStatus.RUNNING

        self.repeat_ct = 0
        return NodeStatus.SUCCESS # TODO handle skips?
