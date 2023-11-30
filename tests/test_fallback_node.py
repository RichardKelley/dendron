from dendron.actions import AlwaysFailureNode, AlwaysSuccessNode
from dendron.controls import FallbackNode

from dendron.basic_types import NodeStatus

def test_failing_fallback():
    n1 = AlwaysFailureNode("Failure1")
    n2 = AlwaysFailureNode("Failure2")

    fallback = FallbackNode("Fallback1", [n1, n2])

    result = fallback.execute_tick()

    assert result == NodeStatus.FAILURE 
