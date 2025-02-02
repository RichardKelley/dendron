from typing import Type, Dict
from .tree_node import TreeNode

# Global registries to store all available classes
node_registry: Dict[str, Type] = {}
config_registry: Dict[str, Type] = {}

def register_dendron_node(cls: Type) -> Type:
    """
    Decorator to register a class in the global registry.
    Can be used with both library and user-defined classes.
    """
    if not issubclass(cls, TreeNode):
        raise TypeError(f"Class {cls.__name__} must be a subclass of TreeNode")
    if cls.__name__ in node_registry:
        raise ValueError(f"Class {cls.__name__} is already registered")
    node_registry[cls.__name__] = cls
    return cls

def register_external_node(cls: Type):
    """
    Decorator to register a class in the global registry.
    Can be used with both library and user-defined classes.
    """
    if not issubclass(cls, TreeNode):
        raise TypeError(f"Class {cls.__name__} must be a subclass of TreeNode")
    if cls.__name__ in node_registry:
        raise ValueError(f"Class {cls.__name__} is already registered")
    node_registry[cls.__name__] = cls

def register_config(cls: Type) -> Type:
    """
    Decorator to register a config class in the config registry.
    """
    if cls.__name__ in config_registry:
        raise ValueError(f"Config {cls.__name__} is already registered")
    config_registry[cls.__name__] = cls
    return cls 