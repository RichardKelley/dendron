from dendron import *

def test_logging0(capsys):

    factory = BehaviorTreeFactory()
    factory.register_simple_action("Action1", lambda: NodeStatus.FAILURE)
    factory.register_simple_action("Action2", lambda: NodeStatus.FAILURE)
    factory.register_simple_condition("AtGoal", lambda: NodeStatus.FAILURE)
    factory.register_simple_action("PortedAction1", lambda: NodeStatus.FAILURE)
    factory.register_simple_condition("Precond1", lambda: NodeStatus.FAILURE)    
    factory.register_simple_condition("Precond2", lambda: NodeStatus.FAILURE)

    tree = factory.create_from_groot("tests/data/TestTree0.xml")

    tree.enable_logging()
    tree.logger.info("test output")

    captured = capsys.readouterr()

    assert "test output" in captured.err
