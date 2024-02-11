from dendron.actions import SimpleAction
from dendron import NodeStatus
from dendron import Blackboard, BlackboardEntryMetadata

def test_simple_action_function():

    person = "Human"
    def simple_action():
        print(f"Hello {person}!")
        return NodeStatus.SUCCESS

    a1 = SimpleAction("greeting", simple_action)
    result = a1.execute_tick()
    assert result == NodeStatus.SUCCESS

def test_simple_action_functor():
    # As in the C++ sense of "functor."

    class Greeter:
        def __init__(self, name):
            self.name = name

        def __call__(self):
            print(f"Hello, {self.name}!")
            return NodeStatus.SUCCESS

    world = Greeter("World")
    a1 = SimpleAction("greeting", world)
    result = a1.execute_tick()
    assert result == NodeStatus.SUCCESS

def test_simple_action_blackboard_update():
    bb = Blackboard()
    bb["important_number"] = 41

    class Updater:
        def __init__(self):
            self.bb = None

        def set_bb(self, bb):
            self.bb = bb

        def __call__(self):
            if self.bb is not None:
                self.bb["important_number"] = 42
                return NodeStatus.SUCCESS
            else:
                return NodeStatus.FAILURE

    action = Updater()
    node = SimpleAction("update_bb", action)
    node.callback.set_bb(bb)

    result = node.execute_tick()
    assert result == NodeStatus.SUCCESS
    assert bb["important_number"] == 42
        