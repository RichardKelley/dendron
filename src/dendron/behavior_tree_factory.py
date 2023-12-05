from .actions import SimpleActionNode

from .blackboard import Blackboard
from .behavior_tree import BehaviorTree

import xml.etree.ElementTree as ET 

class BehaviorTreeFactory:

    def __init__(self):
        self.registry = {}

    def register_action_type(self, name, action):
        pass

    def register_condition_type(self, name, condition):
        pass

    def register_decorator_type(self, name, decorator):
        pass

    def register_simple_action(self, name, action_function):
        pass

    def register_simple_condition(self, name, condition_function):
        pass

    def create_from_xml(self, xml_filename):
        bt = BehaviorTree()

        xml_tree = ET.parse(xml_filename)

        # TODO

        return bt

    