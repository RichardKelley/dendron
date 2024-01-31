from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..condition_node import ConditionNode

from typing import List

class ConjunctionNode(ConditionNode):
    """
    EXPERIMENTAL! DO NOT USE!

    A conjunction is logically equivalent to a sequence node. The
    (eventual) idea is that this will be a node whose children
    can be reordered dynamically as the tree learns.
    """

    def __init__(self, name, children : List[TreeNode] = []) -> None:
        super().__init__(name, children)

        self.current_child_idx = 0

    def tick(self) -> NodeStatus:
        n_children = self.children_count()
        self.set_status(NodeStatus.RUNNING)

        while(self.current_child_idx < n_children):
            current_child = self.children[self.current_child_idx]

            child_status = current_child.execute_tick()
            
            match child_status:
                case NodeStatus.RUNNING:
                    return NodeStatus.RUNNING
                case NodeStatus.FAILURE:
                    self.reset_children()
                    self.current_child_idx = 0
                    return child_status
                case NodeStatus.SUCCESS:
                    self.current_child_idx += 1
                case NodeStatus.SKIPPED:
                    self.current_child_idx += 1
                case NodeStatus.IDLE:
                    raise RuntimeError("Child can't return IDLE")

        if self.current_child_idx == n_children:
            self.reset_children()
            self.current_child_idx = 0

        return NodeStatus.SUCCESS
