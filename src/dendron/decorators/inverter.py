from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class Inverter(DecoratorNode):
    """
    An Inverter decorator instructs its child node to `tick()` and
    then returns the negation of the child's status as its own.

    Args:
        name (`str`):
            The given name of this node.
        child (`dendron.tree_node.TreeNode`):
            Optional child node. If `None`, it is the responsibility of
            the caller to ensure that the `child_node` member variable
            is set before the first `tick()` call.
    """
    def __init__(self, name, child : TreeNode = None) -> None:
        super().__init__(name)
        self.name = name
        self.child_node = child 

    def tick(self) -> NodeStatus:
        """
        Instruct the child node to execute its `tick()` function, and then 
        return `SUCCESS` if the child fails, and `FAILURE` if the child 
        succeeds. Returns `RUNNING` if the child returns `RUNNING`.
        """
        self.set_status(NodeStatus.RUNNING)

        child_status = self.child_node.execute_tick()

        match child_status:
            case NodeStatus.SUCCESS:
                self.reset()
                return NodeStatus.FAILURE
            case NodeStatus.FAILURE:
                self.reset()
                return NodeStatus.SUCCESS
            case NodeStatus.IDLE:
                raise RuntimeError("Child can't return IDLE")
            case _:
                return child_status