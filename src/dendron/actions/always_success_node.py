from ..action_node import ActionNode
from ..tree_node import NodeConfig, NodeStatus
from ..blackboard import Blackboard

class AlwaysSuccessNode(ActionNode):

    def __init__(self):        
        cfg = NodeConfig(Blackboard(), {}, {}, 0, "/always_success")
        super().__init__("AlwaysSuccess", cfg)

    def tick(self):
        return NodeStatus.SUCCESS
