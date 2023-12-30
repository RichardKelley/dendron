from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode
from .blackboard import Blackboard 

from typing import Optional

import logging

class BehaviorTree:
    def __init__(self, tree_name : str, root_node : TreeNode, bb : Blackboard = None):
        self.tree_name = tree_name
        self.root = root_node

        if bb is None:
            self.blackboard = Blackboard()
        else:
            self.blackboard = bb
        
        self.root.set_blackboard(self.blackboard)

        self.logger = None
        self.log_file_name = None

    def __del__(self):
        self.disable_logging()
    
    def enable_logging(self):
        if self.logger is None:
            self.logger = logging.getLogger(self.tree_name)
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.root.set_logger(self.logger)

    def disable_logging(self):
        if self.logger is not None:
            for h in self.logger.handlers:
                h.close()
                self.logger.removeHandler(h)
        self.logger = None
        self.log_file_name = None

    def set_log_level(self, log_level):
        level_to_set = None
        if type(log_level) == str:
            lvl = log_level.upper()
            match lvl:
                case "DEBUG":
                    level_to_set = logging.DEBUG
                case "INFO":
                    level_to_set = logging.INFO
                case "WARNING":
                    level_to_set = logging.WARNING
                case "ERROR":
                    level_to_set = logging.ERROR
                case "CRITICAL":
                    level_to_set = logging.CRITICAL
        elif type(log_level) == int:
            level_to_set = log_level
        else:
            raise TypeError("log_level must be either int or str")

        if self.logger is not None:
            self.logger.setLevel(level_to_set)
            for h in self.logger.handlers:
                h.setLevel(level_to_set)
            self.root.set_log_level(level_to_set)

    def set_log_filename(self, filename : Optional[str]):
        if self.logger is not None:
            for h in self.logger.handlers:
                h.close()
                self.logger.removeHandler(h)
            
            log_level = self.logger.level
            f = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")            

            if filename is None:
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                handler.setFormatter(f)
                self.logger.addHandler(handler)
            else:
                handler = logging.FileHandler(filename)
                handler.setLevel(log_level)
                handler.setFormatter(f)
                self.logger.addHandler(handler)                

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