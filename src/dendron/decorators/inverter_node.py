from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class InverterNode(DecoratorNode):
    def __init__(self, name, child : TreeNode = None):
        self.name = name
        self.child_node = child 

    def tick(self):
        self.set_status(NodeStatus.RUNNING)

        child_status = self.child_node.execute_tick()

        match child_status:
            case NodeStatus.SUCCESS:
                self.reset_child()
                return NodeStatus.FAILURE
            case NodeStatus.FAILURE:
                self.reset_child()
                return NodeStatus.SUCCESS
            case NodeStatus.IDLE:
                raise RuntimeError("Child can't return IDLE")
            case _:
                return child_status