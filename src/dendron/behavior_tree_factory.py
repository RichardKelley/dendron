from .action_node import ActionNode
from .condition_node import ConditionNode
from .control_node import ControlNode
from .decorator_node import DecoratorNode
from .controls import Fallback, Sequence
from .actions import (
    AlwaysFailure, 
    AlwaysSuccess, 
    SimpleAction, 
    AsyncAction, 
    CausalLMAction, 
    ImageLMAction, 
    PipelineAction
)
from .decorators import Inverter
from .conditions import SimpleCondition
from .blackboard import Blackboard
from .behavior_tree import BehaviorTree
from .basic_types import NodeType
from .tree_node import TreeNode
from .xml_utilities import cycle_free, get_parse_order

import xml.etree.ElementTree as ET 
from copy import deepcopy

class BehaviorTreeFactory:
    """
    A factory for behavior trees. This allows the registration of new
    node types and the generation of `BehaviorTree`s from XML files.

    A factory maintains state that allows node repetition and subtree 
    insertion to be automatically handled.
    """

    def __init__(self) -> None:
        self.registry = {}
        self.node_counts = {}
        self.node_types = {}
        self.functors = {}
        self.neural_configs = {}

        self.registry["Fallback"] = Fallback
        self.registry["Sequence"] = Sequence
        self.registry["Inverter"] = Inverter
        self.registry["AlwaysSuccess"] = AlwaysSuccess
        self.registry["AlwaysFailure"] = AlwaysFailure
        self.registry["AsyncAction"] = AsyncAction
        self.registry["CausalLMAction"] = CausalLMAction
        self.registry["ImageLMAction"] = ImageLMAction
        self.registry["PipelineAction"] = PipelineAction
        
        # We replace SubTree nodes with the subtree root, so 
        # we use None as a placeholder here. 
        self.registry["SubTree"] = None 

        self.node_counts["Fallback"] = 0
        self.node_counts["Sequence"] = 0
        self.node_counts["Inverter"] = 0
        self.node_counts["AlwaysSuccess"] = 0
        self.node_counts["AlwaysFailure"] = 0
        self.node_counts["AsyncAction"] = 0
        self.node_counts["CausalLMAction"] = 0
        self.node_counts["ImageLMAction"] = 0
        self.node_counts["PipelineAction"] = 0

        self.node_types["Fallback"] = NodeType.CONTROL
        self.node_types["Sequence"] = NodeType.CONTROL
        self.node_types["Inverter"] = NodeType.DECORATOR
        self.node_types["AlwaysSuccess"] = NodeType.ACTION
        self.node_types["AlwaysFailure"] = NodeType.ACTION
        self.node_types["AsyncAction"] = NodeType.ACTION
        self.node_types["CausalLMAction"] = NodeType.ACTION
        self.node_types["ImageLMAction"] = NodeType.ACTION
        self.node_types["PipelineAction"] = NodeType.ACTION

        self.node_types["SubTree"] = NodeType.SUBTREE

        self.current_blackboard = None
        self.tree_nodes_model = None
        self.behavior_trees = {}

    def register_neural_config(self, name, cfg) -> None:
        """
        Register a configuration object for a neural network
        based node.

        Args:
            name (str):
                The name of the configuration object. This should match
                the name used in Groot.
            cfg:
                The configuration object. At present, one of `CausalLMActionConfig`,
                `ImageLMActionConfig`, `PipelineActionConfig`, or `CompletionConditionConfig`.
        """
        self.neural_configs[name] = cfg

    def register_action_type(self, name, action) -> None:
        """
        Register a new type of action node.

        Args:
            name (str):
                The name of the new action node type.
            action:
                The constructor (class name) of the new node type.
        """
        self.registry[name] = action 
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.ACTION

    def register_condition_type(self, name, condition) -> None:
        """
        Register a new type of condition node.

        Args:
            name (str):
                The name of the new condition type.
            condition:
                The constructor (class name) of the new node type.
        """
        self.registry[name] = condition
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.CONDITION

    def register_decorator_type(self, name, decorator) -> None:
        """
        Register a new type of decorator node.

        Args:
            name (str):
                The name of the new decorator type.
            decorator:
                The constructor (class name) of the new node type.
        """
        self.registry[name] = decorator
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.DECORATOR

    def register_simple_action(self, name, action_function) -> None:
        """
        Register a new simple action. Allows the specification of an
        action and a callback in one step.

        Args:
            name (str):
                The name of the new simple action node type.
            action_function (Callable):
                A callback to execute each time this node is ticked.
        """
        self.registry[name] = SimpleAction
        self.functors[name] = action_function
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.ACTION

    def register_simple_condition(self, name, condition_function) -> None:
        """
        Register a new simple condition. Allows the specification of
        a condition and a callback in one step.

        Args:
            name (str):
                The name of the simple condition node type.
            condition_function (Callable):
                A callback to execute each time this node is ticked.
        """
        self.registry[name] = SimpleCondition
        self.functors[name] = condition_function
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.CONDITION

    def create_from_groot(self, xml_filename : str) -> BehaviorTree:
        """
        Create a `BehaviorTree` instance from an XML file generated by the 
        open-source Groot2 program.

        Args:
            xml_filename (`str`):
                The name of the file containing the XML.
        
        Returns:
            `BehaviorTree`: A behavior tree that instantiates the structure 
            described in the XML file.
        """
        self.current_blackboard = Blackboard()

        xml_tree = ET.parse(xml_filename)
        xml_root = xml_tree.getroot()

        if not "BTCPP_format" in xml_root.attrib:
            raise RuntimeError("XML missing BTCPP_format")
        if xml_root.attrib["BTCPP_format"] != "4":
            raise RuntimeError("BTCPP_format must be 4")

        has_main_tree = "main_tree_to_execute" in xml_root.attrib
        main_tree_name = None
        if has_main_tree:
            main_tree_name = xml_root.attrib["main_tree_to_execute"]
        
        tree_nodes_xml = None
        behavior_tree_xml = []
        main_tree = None
        for child in xml_root:
            if child.tag == "TreeNodesModel":
                tree_nodes_xml = child
            elif child.tag == "BehaviorTree":
                if "ID" in child.attrib and child.attrib["ID"] == main_tree_name:
                    main_tree = child
                else:
                    behavior_tree_xml.append(child)

        # load TreeNodesModel
        for child in tree_nodes_xml:
            match child.tag:
                case "Action":
                    self.node_types[child.attrib["ID"]] = NodeType.ACTION
                case "Condition":
                    self.node_types[child.attrib["ID"]] = NodeType.CONDITION
                case "Control":
                    self.node_types[child.attrib["ID"]] = NodeType.CONTROL
                case "Decorator":
                    self.node_types[child.attrib["ID"]] = NodeType.DECORATOR

        if not has_main_tree and len(behavior_tree_xml) > 1:
            raise RuntimeError("Multiple behavior trees but no main tree.")

        if main_tree is None and len(behavior_tree_xml) == 1:
            main_tree = behavior_tree_xml[0]

        # load each behavior tree
        ## convert the other trees
        parse_order = get_parse_order(xml_root)
        for tree_name in parse_order:
            if tree_name == main_tree_name:
                continue
            tree = None
            for child in xml_root:
                if child.tag != "BehaviorTree":
                    continue
                child_name = child.attrib["ID"]
                if child_name != tree_name:
                    continue
                else:
                    tree = self.parse_behavior_tree_groot(child_name, child)
                    self.behavior_trees[tree_name] = tree

        # parse the main tree last
        main_tree = self.parse_behavior_tree_groot(main_tree.attrib["ID"], main_tree)
        self.behavior_trees[main_tree_name] = main_tree

        return main_tree
        
    def parse_behavior_tree_groot(self, tree_name, xml_node) -> BehaviorTree:
        tree_type = self.node_types[xml_node[0].tag]
        root_node = None
        match tree_type:
            case NodeType.ACTION:
                root_node = self.parse_action_node_groot(xml_node[0])
            case NodeType.CONDITION:
                root_node = self.parse_condition_node_groot(xml_node[0])
            case NodeType.CONTROL:
                root_node = self.parse_control_node_groot(xml_node[0])
            case NodeType.DECORATOR:
                root_node = self.parse_decorator_node_groot(xml_node[0])
            case NodeType.SUBTREE:
                root_node = self.parse_subtree_node_groot(xml_node[0])

        bt = BehaviorTree(tree_name, root_node)
        return bt

    def parse_action_node_groot(self, xml_node) -> ActionNode:
        tag = xml_node.tag
        if not tag in self.registry:
            raise KeyError(f"Undefined action {tag}.")

        node_id = self.node_counts[tag]
        node_name = tag + "_" + str(node_id)
    
        if self.registry[tag] == SimpleAction:
            f = self.functors[tag]
            new_node = self.registry[tag](node_name, f)
        else:
            new_node = self.registry[tag](node_name)

        self.node_counts[tag] += 1
        
        if xml_node.attrib:
            for key in xml_node.attrib:
                self.current_blackboard[key] = xml_node.attrib[key]
                
        return new_node

    def parse_condition_node_groot(self, xml_node) -> ConditionNode:
        tag = xml_node.tag
        if not tag in self.registry:
            raise KeyError(f"Undefined condition {tag}.")

        node_id = self.node_counts[tag]
        node_name = tag + "_" + str(node_id)

        if self.registry[tag] == SimpleCondition:
            f = self.functors[tag]
            new_node = self.registry[tag](node_name, f)
        else:
            new_node = self.registry[tag](node_name)

        self.node_counts[tag] += 1

        if xml_node.attrib:
            for key in xml_node.attrib:
                self.current_blackboard[key] = xml_node.attrib[key]

        return new_node

    def parse_control_node_groot(self, xml_node) -> ControlNode:
        tag = xml_node.tag
        if not tag in self.registry:
            raise KeyError(f"Undefined control {tag}.")

        node_id = self.node_counts[tag]
        node_name = tag + "_" + str(node_id)

        self.node_counts[tag] += 1

        if xml_node.attrib:
            for key in xml_node.attrib:
                self.current_blackboard[key] = xml_node.attrib[key]

        # parse children
        child_nodes = []
        for child_xml in xml_node:
            if not child_xml.tag in self.registry:
                raise RuntimeError(f"Unregistered node {child_xml.tag}")

            match self.node_types[child_xml.tag]:
                case NodeType.ACTION:
                    child_node = self.parse_action_node_groot(child_xml)
                case NodeType.CONDITION:
                    child_node = self.parse_condition_node_groot(child_xml)
                case NodeType.CONTROL:
                    child_node = self.parse_control_node_groot(child_xml)
                case NodeType.DECORATOR:
                    child_node = self.parse_decorator_node_groot(child_xml)
                case NodeType.SUBTREE:
                    child_node = self.parse_subtree_node_groot(child_xml)

            child_nodes.append(child_node)

        new_node = self.registry[tag](node_name, child_nodes)

        return new_node

    def parse_decorator_node_groot(self, xml_node) -> DecoratorNode:
        tag = xml_node.tag
        if not tag in self.registry:
            raise KeyError(f"Undefined decorator {tag}.")

        node_id = self.node_counts[tag]
        node_name = tag + "_" + str(node_id)

        self.node_counts[tag] += 1

        if xml_node.attrib:
            for key in xml_node.attrib:
                self.current_blackboard[key] = xml_node.attrib[key]

        child_node = None
        child_xml = xml_node[0]
        match self.node_types[child_xml.tag]:
            case NodeType.ACTION:
                child_node = self.parse_action_node_groot(child_xml)
            case NodeType.CONDITION:
                child_node = self.parse_condition_node_groot(child_xml)
            case NodeType.CONTROL:
                child_node = self.parse_control_node_groot(child_xml)
            case NodeType.DECORATOR:
                child_node = self.parse_decorator_node_groot(child_xml)
            case NodeType.SUBTREE:
                child_node = self.parse_subtree_node_groot(child_xml)

        new_node = self.registry[tag](node_name, child_node)
        return new_node

    def parse_subtree_node_groot(self, xml_node) -> TreeNode:
        subtree_name = xml_node.attrib["ID"]

        # TODO is deepcopy good enough?
        return deepcopy(self.behavior_trees[subtree_name].root)