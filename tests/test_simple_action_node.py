from dendron import SimpleActionNode
from dendron import NodeStatus

def test_simple_action_function():

    person = "Human"
    def simple_action():
        print(f"Hello {person}!")
        return NodeStatus.SUCCESS

    a1 = SimpleActionNode("greeting", simple_action)
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
    a1 = SimpleActionNode("greeting", world)
    result = a1.execute_tick()
    assert result == NodeStatus.SUCCESS