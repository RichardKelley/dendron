from .action_node import ActionNode
from .condition_node import ConditionNode
from .control_node import ControlNode
from .decorator_node import DecoratorNode

from .controls import FallbackNode, SequenceNode
from .actions import AlwaysFailureNode, AlwaysSuccessNode, SimpleActionNode
from .decorators import InverterNode

from .blackboard import Blackboard
from .behavior_tree import BehaviorTree

from .basic_types import NodeType

import xml.etree.ElementTree as ET 

class BehaviorTreeFactory:

    def __init__(self):
        self.registry = {}
        self.node_counts = {}
        self.node_types = {}
        self.functors = {}

        self.registry["Fallback"] = FallbackNode
        self.registry["Sequence"] = SequenceNode
        self.registry["Inverter"] = InverterNode
        self.registry["AlwaysSuccess"] = AlwaysSuccessNode
        self.registry["AlwaysFailure"] = AlwaysFailureNode

        self.node_counts["Fallback"] = 0
        self.node_counts["Sequence"] = 0
        self.node_counts["Inverter"] = 0
        self.node_counts["AlwaysSuccess"] = 0
        self.node_counts["AlwaysFailure"] = 0

        self.node_types["Fallback"] = NodeType.CONTROL
        self.node_types["Sequence"] = NodeType.CONTROL
        self.node_types["Inverter"] = NodeType.DECORATOR
        self.node_types["AlwaysSuccess"] = NodeType.ACTION
        self.node_types["AlwaysFailure"] = NodeType.ACTION

        self.current_blackboard = None
        self.tree_nodes_model = None

    def register_action_type(self, name, action):
        self.registry[name] = action 
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.ACTION

    def register_condition_type(self, name, condition):
        self.registry[name] = condition
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.CONDITION

    def register_decorator_type(self, name, decorator):
        self.registry[name] = decorator
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.DECORATOR

    def register_simple_action(self, name, action_function):
        self.registry[name] = SimpleActionNode
        self.functors[name] = action_function
        self.node_counts[name] = 0
        self.node_types[name] = NodeType.ACTION

    def register_simple_condition(self, name, condition_function):
        # TODO
        pass 

    def create_from_xml(self, xml_filename):
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

        # load TreeNodesModel
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

        if not has_main_tree and len(behavior_tree_xml) > 1:
            raise RuntimeError("Multiple behavior trees but no main tree.")

        if main_tree is None and len(behavior_tree_xml) == 1:
            main_tree = behavior_tree_xml[0]

        # load each behavior tree
        ## convert the main tree
        main_tree_type = self.node_types[main_tree[0].tag]
        root_node = None
        match main_tree_type:
            case NodeType.ACTION:
                root_node = self.parse_action_node_xml(main_tree[0])
            case NodeType.CONDITION:
                root_node = self.parse_condition_node_xml(main_tree[0])
            case NodeType.CONTROL:
                root_node = self.parse_control_node_xml(main_tree[0])
            case NodeType.DECORATOR:
                root_node = self.parse_decorator_node_xml(main_tree[0])

        ## convert the other trees
        # TODO

        bt = BehaviorTree(root_node)
        return bt

    def parse_tree_nodes_model_xml(self, xml_node):
        if len(xml_node) == 0:
            return
        for child in xml_node:
            new_node_type = None
            match child.tag:
                case "Action":
                    print(f"Adding action {child['ID']}")
                case "Condition":
                    print(f"Adding condition {child['ID']}")
                case "Control":
                    print(f"Adding control {child['ID']}")
                case "Decorator":
                    print(f"Adding decorator {child['ID']}")
        

    def parse_action_node_xml(self, xml_node) -> ActionNode:
        tag = xml_node.tag
        if not tag in self.registry:
            raise KeyError(f"Undefined action {tag}.")

        node_id = self.node_counts[tag]
        node_name = tag + "_" + str(node_id)
    
        if self.registry[tag] == SimpleActionNode:
            f = self.functors[tag]
            new_node = self.registry[tag](node_name, f)
        else:
            new_node = self.registry[tag](node_name)

        self.node_counts[tag] += 1
        
        if xml_node.attrib:
            for key in xml_node.attrib:
                self.current_blackboard[key] = xml_node.attrib[key]
                
        return new_node

    def parse_condition_node_xml(self, xml_node) -> ConditionNode:
        tag = xml_node.tag
        if not tag in self.registry:
            raise KeyError(f"Undefined condition {tag}.")

        node_id = self.node_counts[tag]
        node_name = tag + "_" + str(node_id)

        new_node = self.registry[tag](node_name)

        self.node_counts[tag] += 1

        if xml_node.attrib:
            for key in xml_node.attrib:
                self.current_blackboard[key] = xml_node.attrib[key]

        return new_node

    def parse_control_node_xml(self, xml_node) -> ControlNode:
        tag = xml_node.tag
        if not tag in self.registry:
            raise KeyError(f"Undefined control {tag}.")

        node_id = self.node_counts[tag]
        node_name = tag + "_" + str(node_id)

        new_node = self.registry[tag](node_name)

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
                    child_node = self.parse_action_node_xml(child_xml)
                case NodeType.CONDITION:
                    child_node = self.parse_condition_node_xml(child_xml)
                case NodeType.CONTROL:
                    child_node = self.parse_control_node_xml(child_xml)
                case NodeType.DECORATOR:
                    child_node = self.parse_decorator_node_xml(child_xml)

            child_nodes.append(child_node)

        new_node.add_children(child_nodes)
        return new_node

    def parse_decorator_node_xml(self, xml_node) -> DecoratorNode:
        pass
    