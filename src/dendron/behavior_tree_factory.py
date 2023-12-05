from .actions import SimpleActionNode

from .action_node import ActionNode
from .condition_node import ConditionNode
from .control_node import ControlNode
from .decorator_node import DecoratorNode

from .blackboard import Blackboard
from .behavior_tree import BehaviorTree

import xml.etree.ElementTree as ET 

class BehaviorTreeFactory:

    def __init__(self):
        self.registry = {}

    def register_action_type(self, name, action):
        self.registry[name] = action 

    def register_condition_type(self, name, condition):
        self.registry[name] = condition

    def register_decorator_type(self, name, decorator):
        self.registry[name] = decorator

    def register_simple_action(self, name, action_function):
        simple_action = SimpleActionNode("name", action_function)
        self.registry[name] = simple_action

    def register_simple_condition(self, name, condition_function):
        # TODO
        pass 

    def create_from_xml(self, xml_filename):
        bt = BehaviorTree()

        xml_tree = ET.parse(xml_filename)

        # TODO

        return bt

    def parse_action_node_xml(self, xml_node) -> ActionNode:
        pass

    def parse_condition_node_xml(self, xml_node) -> ConditionNode:
        pass

    def parse_control_node_xml(self, xml_node) -> ControlNode:
        pass

    def parse_decorator_node_xml(self, xml_node) -> DecoratorNode:
        pass
    