from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..control_node import ControlNode

from typing import List 

class Fallback(ControlNode):
    """
    A Fallback node is a control node that ticks its children in 
    sequence, until a child returns `SUCCESS`, at which point it
    returns success. If all children return `FAILURE`, then the
    Fallback node returns `FAILURE`. The intuition is that the
    Fallback node is trying different options until it finds one
    that works.

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
        status of `SUCCESS`. If all children fail, return `FAILURE`.

        Returns:
            `NodeStatus`: `SUCCESS` if at least one child succeeds,
            `FAILURE` otherwise. May return `RUNNING` or `SKIPPED`
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
                    self.current_child_idx += 1
                case NodeStatus.SUCCESS:
                    self.reset()
                    self.current_child_idx = 0
                    return child_status
                case NodeStatus.SKIPPED:
                    self.current_child_idx += 1
                case NodeStatus.IDLE:
                    raise RuntimeError("Child can't return IDLE")
            
        if self.current_child_idx == n_children:
            self.reset()

        return NodeStatus.FAILURE

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
        repr = f"{tabs}Fallback {self.name}"
        for child in self.children:
            child_repr = child.pretty_repr(depth+1)
            repr += f"\n{child_repr}"
        repr += "\n"
        return repr