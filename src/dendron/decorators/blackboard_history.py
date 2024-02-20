from ..blackboard import Blackboard
from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class BlackboardHistory(DecoratorNode):
    """
    The Blackboard history node keeps track of a blackboard entry 
    related to a child node. Every time this node is ticked, it 
    examines the blackboard and records the value stored at the
    `child_key` before `tick()`ing the child.

    The history is itself stored in the blackboard, by default at
    the key "{child_node.name}/{child_key}/history".

    Args:
        name (`str`):
            The given name of this node.
        child (`dendron.tree_node.TreeNode`):
            The child node whose blackboard history we want to track.
        child_key (`str`):
            The blackboard key we want to record values for.
    """
    def __init__(self, name, child: TreeNode, child_key : str = "in") -> None:
        super().__init__(name, child)

        self.child_key = self.child_node.input_key
        self.history_key = f"{self.child_node.name}/{child_key}/history"
        if self.blackboard is not None:
            self.blackboard[self.history_key] = []

    def set_blackboard(self, bb : Blackboard) -> None:
        """
        Assign a new blackboard for history tracking.

        Args:
            bb (`dendron.blackboard.Blackboard`):
                The new blackboard to track.
        """
        self.blackboard = bb
        self.blackboard[self.history_key] = []
        
        self.child_node.set_blackboard(bb)

    def reset(self) -> None:
        """
        Clear the history and instruct the child to reset.
        """
        self.blackboard[self.history_key] = []
        self.child_node.reset()

    def tick(self) -> NodeStatus:
        """
        Record the value stored in the blackboard at `child_key` and
        then instruct the child node to execute its `tick()` function.
        """
        latest = self.blackboard[self.child_key]
        self.blackboard[self.history_key].append(latest)

        status = self.child_node.execute_tick()

        return status
