from dendron.actions import AlwaysSuccess
from dendron.controls import Sequence
from dendron.basic_types import NodeStatus

def test_successful_sequence():
    n1 = AlwaysSuccess("Success1")
    n2 = AlwaysSuccess("Success2")

    seq = Sequence(name="Sequence")

    seq.add_child(n1)
    seq.add_child(n2)

    result = seq.execute_tick()

    assert result == NodeStatus.SUCCESS