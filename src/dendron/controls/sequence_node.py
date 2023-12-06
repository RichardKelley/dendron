from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..control_node import ControlNode

from typing import List

class SequenceNode(ControlNode):

    def __init__(self, name, children : List[TreeNode] = []):
        super().__init__(name, children)

        self.current_child_idx = 0


    def halt_node(self):
        self.current_child_idx = 0
        ControlNode.halt(self)

    def tick(self):
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

    def pretty_repr(self, depth = 0):
        tabs = '\t'*depth
        repr = f"{tabs}Sequence {self.name}"
        for child in self.children:
            child_repr = child.pretty_repr(depth+1)
            repr += f"\n{child_repr}"
        repr += "\n"
        return repr