"""
Dendron is a library for building behavior trees that use
large language models and vision language models.
"""

__version__ = "0.2.0"
__author__ = "Richard Kelley"

from .registry import class_registry, register_class, config_registry, register_config
from .basic_types import NodeStatus
from .behavior_tree import BehaviorTree 
from .blackboard import Blackboard

def initialize_registry():
    """Initialize the global class registry by importing all modules with decorated classes"""
    from dendron.actions import (
        SimpleAction, AsyncAction,
        GenerateAction, PipelineAction, LogLikelihoodAction,
        LogLikelihoodRollingAction
    )
    from dendron.conditions import (
        LMCompletionCondition
    )
    from dendron.decorators import (
        Inverter, ForceSuccess, RunOnce, BlackboardHistory, Repeat, Retry
    )
    from dendron.controls import (
        Fallback, Sequence
    )
    from dendron.configs import (
        HFLMConfig, LMActionConfig, LMCompletionConfig
    )

initialize_registry()