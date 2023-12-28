from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode
from .blackboard import Blackboard 

from typing import Optional

class BehaviorTree:
    def __init__(self, root_node : TreeNode, bb : Blackboard = None):
        self.root = root_node

        if bb is None:
            self.blackboard = Blackboard()
        else:
            self.blackboard = bb
        
        self.root.set_blackboard(self.blackboard)
            
    def set_root(self, new_root):
        self.root = new_root

    def halt_tree(self):
        self.root.halt_node()

    def blackboard_get(self, key):
        return self.blackboard[key]

    def blackboard_set(self, key, value):
        self.blackboard[key] = value

    def get_node_by_name(self, name : str) -> Optional[TreeNode]:
        if self.root:
            return self.root.get_node_by_name(name)
        else:
            return None

    def tick_once(self):
        return self.root.execute_tick()

    def tick_while_running(self):
        status = self.root.execute_tick()
        while status == NodeStatus.RUNNING:
            status = self.root.execute_tick()

    def pretty_print(self):
        print(self.root.pretty_repr())