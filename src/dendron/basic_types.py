from enum import Enum

class NodeType(Enum):
    """
    Enum containing the types of nodes allowed in the behavior tree
    framework. Some of these are experimental.
    """
    UNDEFINED = 0
    ACTION = 1
    CONDITION = 2
    CONTROL = 3
    DECORATOR = 4
    GOAL = 5
    CONJUNCTION = 6
    DISJUNCTION = 7
    SUBTREE = 8

class NodeStatus(Enum):
    """
    Enum containing the allowable return values from node `tick` 
    functions.
    """
    IDLE = 0
    RUNNING = 1
    SUCCESS = 2
    FAILURE = 3
    SKIPPED = 4

class Quantization(Enum):
    """
    Enum representing currently allowable quantization levels for
    neural models. `TwoBit` is currently aspirational.
    """
    NoQuantization = 0, 
    TwoBit = 2,
    FourBit = 4,
    EightBit = 8,