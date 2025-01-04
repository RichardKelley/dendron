from .basic_types import NodeType, NodeStatus, ModelConfig
from .tree_node import TreeNode
from .blackboard import Blackboard 

from hflm import LM, HFLM

from typing import Optional, Any

import logging

from concurrent import futures

class BehaviorTree:
    """
    A `BehaviorTree` instance is a container for the nodes that make
    up a behavior tree. This object is responsible for maintaining a
    root node of the tree, a blackboard that is shared among the nodes
    of the tree, and a thread pool for asynchronous action nodes. 

    Args:
        tree_name (`str`):
            The given name of this tree.
        root_node (`dendron.tree_node.TreeNode`):
            The root node of this tree.
        bb (`dendron.blackboard.Blackboard`):
            An optional pre-initialized blackboard to use in this tree.
        num_workers (`int`):
            An optional number of workings to initialize the thread pool
            with.
    """
    def __init__(self, tree_name : str, root_node : TreeNode = None, bb : Blackboard = None, num_workers=4) -> None:
        self.tree_name = tree_name
        self.root = root_node

        if bb is None:
            self.blackboard = Blackboard()
        else:
            self.blackboard = bb
        
        if self.root is not None:
            self.root.set_blackboard(self.blackboard)
            self.root.set_tree(self)

        self.num_workers = num_workers
        self.logger = None
        self.log_file_name = None

        # Mapping from model names to config objects.
        self.model_configs = {}

        # Mapping from model names to HFLM model objects.
        self.models = {}

        self.executor = futures.ThreadPoolExecutor(max_workers=num_workers)

    def get_config(self, model_name : str) -> Optional[ModelConfig]:
        if model_name in self.model_configs:
            return self.model_configs[model_name]
        else:
            return None
    
    def get_model(self, model_name : str) -> Optional[LM]:
        if model_name in self.models:
            return self.models[model_name]
        else:
            return None
        
    def add_model(self, model_config : ModelConfig) -> None:
        """
        Add a model to the behavior tree's model registry.
        
        Args:
            model_config (ModelConfig):
                Configuration object containing model parameters
        """
        if model_config.model_name not in self.model_configs:
            name = model_config.model_name
            self.model_configs[name] = model_config
            self.models[name] = HFLM(
                model=model_config.model,
                device=model_config.device,
                parallelize=model_config.parallelize,
                load_in_4bit=model_config.load_in_4bit,
                load_in_8bit=model_config.load_in_8bit
            )

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
        """
        Turn on logging for every node in this tree. By default,
        each `tick()` call in every node results in a logging event.
        """
        if self.logger is None:
            self.logger = logging.getLogger(self.tree_name)
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            if self.root is not None:
                self.root.set_logger(self.logger)

    def disable_logging(self) -> None:
        """
        Turn logging off. 
        """
        if self.logger is not None:
            for h in self.logger.handlers:
                h.close()
                self.logger.removeHandler(h)
        self.logger = None
        self.log_file_name = None
        if self.root is not None:
            self.root.set_logger(None)

    def set_log_level(self, log_level) -> None:
        """
        Set the log level for the tree. This is a no-op if logging
        is not enabled.
        """
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

            if self.root is not None:
                self.root.set_log_level(level_to_set)

    def set_log_filename(self, filename : Optional[str]) -> None:
        """
        If we want to log to a file instead of the command line, we use
        this method to set a a file name. 

        Alternatively, if we are logging to a file and want to log to a
        stream instead, we can call this method with the filename set to
        `None`.

        Args:
            filename (`Optional[str]`):
                If `None`, log to a stream. If a `filename`, log to a file
                with that name.
        """
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

    def set_root(self, new_root : TreeNode) -> None:
        """
        Set the root of the tree to a new node.

        Args:
            new_root (`dendron.tree_node.TreeNode`):
                The new root node.
        """
        self.root = new_root

        if self.logger is not None:
            self.root.set_logger(self.logger)
            self.root.set_log_level(self.logger.level)

        new_root.set_tree(self)

    def status(self) -> Optional[NodeStatus]:
        """
        Return the current status of this tree. The status of a tree
        is the current status of the root node of hte tree.

        Returns:
            `NodeStatus`: The status of the tree's root.
        """
        if self.root is not None:
            return self.root.get_status()
        else:
            return None 

    def reset(self) -> None:
        """
        Instruct the root of the tree to `reset()`.
        """
        if self.root is not None:
            self.root.reset()

    def halt_tree(self) -> None:
        """
        Instruct the root of the tree to `halt()`.
        """
        if self.root is not None:
            self.root.halt_node()

    # TODO: This was a hallucination. Need to decide if we want to add dependency.
    #@deprecated("Use blackboard directly.")
    def blackboard_get(self, key) -> Any:
        return self.blackboard[key]

    # TODO: This was a hallucination. Need to decide if we want to add dependency.
    #@deprecated("Use blackboard directly.")
    def blackboard_set(self, key, value) -> None:
        self.blackboard[key] = value

    def get_node_by_name(self, name : str) -> Optional[TreeNode]:
        """
        Search for a node by its name. Forwards the call to the current
        root node.

        Args:
            name (`str`):
                The name of the node we are looking for.

        Returns:
            `Optional[TreeNode]`: Either a node with the given name,
            or None.
        """
        if self.root is not None:
            return self.root.get_node_by_name(name)
        else:
            return None

    def tick_once(self) -> Optional[NodeStatus]:
        """
        Instruct the root of the tree to execute its `tick()` function.
        
        This is the primary interface to run a `BehaviorTree`.

        Returns:
            `NodeStatus`: The status returned by the root.
        """
        if self.root is not None:
            return self.root.execute_tick()
        else:
            return None

    def tick_while_running(self) -> Optional[NodeStatus]:
        """
        Repeatedly `tick()` the behavior tree as long as the status
        returned by the root is `RUNNING`. 

        At present, this is only possible if the tree contains one or
        more asynchronous nodes.

        Returns:
            `NodeStatus`: The status ultimately returned by the root.
        """
        if self.root is not None:
            status = self.root.execute_tick()
            while status == NodeStatus.RUNNING:
                status = self.root.execute_tick()
            return status
        else:
            return None

    def pretty_print(self) -> None:
        """
        Print an indented version of this tree to the command line. 
        Indentation shows structure.
        """
        if self.root is not None:
            print(self.root.pretty_repr())
        else:
            print("BehaviorTree{}")