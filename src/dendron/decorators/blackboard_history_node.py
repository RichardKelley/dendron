from ..blackboard import Blackboard
from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class BlackboardHistoryNode(DecoratorNode):
    def __init__(self, name, child: TreeNode, child_key : str = "in") -> None:
        super().__init__(name, child)

        self.child_key = self.child_node.input_key
        self.history_key = f"{self.child_node.name}/{child_key}/history"
        if self.blackboard is not None:
            self.blackboard[self.history_key] = []

    def set_blackboard(self, bb : Blackboard) -> None:
        self.blackboard = bb
        self.blackboard[self.history_key] = []
        
        self.child_node.set_blackboard(bb)

    def reset(self) -> None:
        self.blackboard[self.history_key] = []
        self.child_node.reset()

    def tick(self) -> NodeStatus:
        latest = self.blackboard[self.child_key]
        self.blackboard[self.history_key].append(latest)

        status = self.child_node.execute_tick()

        return status
