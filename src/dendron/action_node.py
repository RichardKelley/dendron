from .basic_types import NodeType, NodeStatus
from .tree_node import NodeConfig, TreeNode

class ActionNode(TreeNode):
    def __init__(self, name, cfg):
        super().__init__(name, cfg)

    def node_type(self) -> NodeType:
        return NodeType.ACTION
        
    #def tick(self):
    #    raise NotImplementedError()

    
