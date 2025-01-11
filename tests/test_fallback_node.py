from dendron.actions import AlwaysFailure, AlwaysSuccess
from dendron.controls import Fallback

from dendron.basic_types import NodeStatus

def test_failing_fallback():
    n1 = AlwaysFailure("Failure1")
    n2 = AlwaysFailure("Failure2")

    fallback = Fallback([n1, n2])

    result = fallback.execute_tick()

    assert result == NodeStatus.FAILURE 
