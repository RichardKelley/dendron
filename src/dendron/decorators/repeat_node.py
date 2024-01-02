from ..basic_types import NodeType, NodeStatus
from ..tree_node import TreeNode
from ..decorator_node import DecoratorNode

class RepeatNode(DecoratorNode):
    def __init__(self, name : str, child : TreeNode, n_times : int):
        super().__init__(name, child)
        self.n_times = n_times
        self.repeat_ct = 0

    def reset(self):
        self.repeat_ct = 0
        self.child_node.reset()

    def tick(self):        
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
