from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus
from dendron.blackboard import Blackboard

def test_action():

    class MyAction(ActionNode):
        def __init__(self, name):
            super().__init__(name)

        def tick(self):
            self.blackboard["input"] = 42
            return NodeStatus.SUCCESS

    bb = Blackboard()
    bb["input"] = 4

    assert bb["input"] == 4

    my_action = MyAction("MyAction")
    my_action.set_blackboard(bb)

    result = my_action.execute_tick()

    assert result == NodeStatus.SUCCESS
    assert bb["input"] == 42