from enum import Enum

class NodeType(Enum):
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
    IDLE = 0
    RUNNING = 1
    SUCCESS = 2
    FAILURE = 3
    SKIPPED = 4

class Quantization(Enum):
    NoQuantization = 0, 
    TwoBit = 2,
    FourBit = 4,
    EightBit = 8,