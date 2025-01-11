from dendron.decorators import Inverter
from dendron.actions import AlwaysSuccess, AlwaysFailure
from dendron.basic_types import NodeStatus

def test_failure_inversion():
    n1 = AlwaysSuccess("SuccessNode")

    inverter = Inverter("Inverter", n1)

    result = inverter.execute_tick()

    assert result == NodeStatus.FAILURE

def test_success_inversion():

    n1 = AlwaysFailure("FailureNode")

    inverter = Inverter("Inverter", n1)

    result = inverter.execute_tick()

    assert result == NodeStatus.SUCCESS