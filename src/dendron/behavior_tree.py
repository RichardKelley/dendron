from .basic_types import NodeType, NodeStatus
from .tree_node import TreeNode
from .blackboard import Blackboard 

from typing import Optional, Any

import logging

from concurrent import futures

class BehaviorTree:
    def __init__(self, tree_name : str, root_node : TreeNode, bb : Blackboard = None, num_workers=4) -> None:
        self.tree_name = tree_name
        self.root = root_node

        if bb is None:
            self.blackboard = Blackboard()
        else:
            self.blackboard = bb
        
        self.root.set_blackboard(self.blackboard)
        self.root.set_tree(self)

        self.num_workers = num_workers
        self.logger = None
        self.log_file_name = None

        self.executor = futures.ThreadPoolExecutor(max_workers=num_workers)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['executor']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.executor = futures.ThreadPoolExecutor(max_workers=self.num_workers)

    def __del__(self):
        self.disable_logging()
    
    def enable_logging(self) -> None:
        if self.logger is None:
            self.logger = logging.getLogger(self.tree_name)
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.root.set_logger(self.logger)

    def disable_logging(self) -> None:
        if self.logger is not None:
            for h in self.logger.handlers:
                h.close()
                self.logger.removeHandler(h)
        self.logger = None
        self.log_file_name = None
        # TODO set root logger to None? 

    def set_log_level(self, log_level) -> None:
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

    def set_log_filename(self, filename : Optional[str]) -> None:
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

    def set_root(self, new_root) -> None:
        self.root = new_root
        new_root.set_tree(self)

    def status(self) -> NodeStatus:
        return self.root.get_status()

    def reset(self) -> None:
        self.root.reset()

    def halt_tree(self) -> None:
        self.root.halt_node()

    # TODO consider deprecating
    def blackboard_get(self, key) -> Any:
        return self.blackboard[key]

    # TODO consider deprecating
    def blackboard_set(self, key, value) -> None:
        self.blackboard[key] = value

    def get_node_by_name(self, name : str) -> Optional[TreeNode]:
        if self.root:
            return self.root.get_node_by_name(name)
        else:
            return None

    def tick_once(self) -> NodeStatus:
        return self.root.execute_tick()

    def tick_while_running(self) -> NodeStatus:
        status = self.root.execute_tick()
        while status == NodeStatus.RUNNING:
            status = self.root.execute_tick()
        return status

    def pretty_print(self) -> None:
        print(self.root.pretty_repr())