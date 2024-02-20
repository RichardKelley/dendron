from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..control_node import ControlNode

from typing import List

class Sequence(ControlNode):
    """
    A Sequence node is a control node that ticks its children in
    sequence, until a child returns `FAILURE`, at which point it
    returns `FAILURE`. If all children succeed, then the Sequence
    node returns `SUCCESS`. 

    Args:
        name (`str`):
            The given name of this node.
        children (`List[TreeNode]`):
            A list of `TreeNode`s to initialize the children of this
            node. Will be ticked in the order they are given.
    """

    def __init__(self, name, children : List[TreeNode] = []) -> None:
        super().__init__(name, children)

        self.current_child_idx = 0

    def reset(self) -> None:
        """
        Set the current child index to 0 and instruct all children
        to reset.
        """
        self.current_child_idx = 0
        for child in self.children:
            child.reset()

    def halt_node(self) -> None:
        """
        Set the current child index to 0 and instruct all children
        to halt via the parent class `halt()`.
        """
        self.current_child_idx = 0
        ControlNode.halt(self)

    def tick(self) -> NodeStatus:
        """
        Successively `tick()` each child node until one returns a 
        status of `FAILURE`. If all children succeed, return 
        `SUCCESS`.

        Returns:
            `NodeStatus`: `FAILURE` if at least one child fails,
            `SUCCESS` otherwise. May return `RUNNING` or `SKIPPED`
            depending on children's behavior.
        """
        n_children = self.children_count()
        self.set_status(NodeStatus.RUNNING)

        while(self.current_child_idx < n_children):
            current_child = self.children[self.current_child_idx]

            child_status = current_child.execute_tick()
            
            match child_status:
                case NodeStatus.RUNNING:
                    return NodeStatus.RUNNING
                case NodeStatus.FAILURE:
                    self.reset()
                    self.current_child_idx = 0
                    return child_status
                case NodeStatus.SUCCESS:
                    self.current_child_idx += 1
                case NodeStatus.SKIPPED:
                    self.current_child_idx += 1
                case NodeStatus.IDLE:
                    raise RuntimeError("Child can't return IDLE")

        if self.current_child_idx == n_children:
            self.reset()

        return NodeStatus.SUCCESS

    def pretty_repr(self, depth = 0) -> str:
        """
        Return a string representation of this node at the given depth.

        Args:
            depth (`int`):
                The depth of this node in a surrounding tree.

        Returns:
            `str`: The indented string representation.
        """
        tabs = '\t'*depth
        repr = f"{tabs}Sequence {self.name}"
        for child in self.children:
            child_repr = child.pretty_repr(depth+1)
            repr += f"\n{child_repr}"
        repr += "\n"
        return repr