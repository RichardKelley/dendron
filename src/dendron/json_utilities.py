import json
from typing import Type, Dict, Any, Optional, List
from abc import ABC

from dendron.behavior_tree import BehaviorTree
from dendron.tree_node import TreeNode
from dendron.configs.hflm_config import HFLMConfig
from dendron.configs.lm_action_config import LMActionConfig
from dendron.configs.lm_completion_config import LMCompletionConfig
from dendron.registry import config_registry, node_registry
from dendron.blackboard import Blackboard


def create_hflm_config(config_data: Dict[str, Any]) -> HFLMConfig:
    """Create an HFLMConfig instance from JSON config data"""
    values = config_data['values']
    
    return HFLMConfig(
        model=values['model_name'],
        device=values['device'],
        parallelize=values['parallelize'],
        dtype=values['dtype'],
        load_in_4bit=values['load_in_4bit'],
        load_in_8bit=values['load_in_8bit'],
        add_bos_token=values['add_bos_token'],
        offload_folder=values['offload_folder']
    )


def create_lm_action_config(config_data: Dict[str, Any]) -> LMActionConfig:
    """Create an LMActionConfig instance from JSON config data"""
    values = config_data['values']
    
    return LMActionConfig(
        node_name=values['node_name'],
        input_key=values['input_key'],
        output_key=values['output_key'],
        completions_key=values['completions_key'],
        max_new_tokens=values['max_new_tokens'],
        temperature=values['temperature'],
        truncation=values['truncation'],
        max_length=values['max_length'],
        prefix_token_id=values['prefix_token_id'],
        batch_size=values['batch_size'],
        max_batch_size=values['max_batch_size']
    )


def create_lm_completion_config(config_data: Dict[str, Any]) -> LMCompletionConfig:
    """Create an LMCompletionConfig instance from JSON config data"""
    values = config_data['values']
    
    return LMCompletionConfig(
        node_name=values['node_name'],
        completions_key=values['completions_key'],
        logprobs_out_key=values['logprobs_out_key'],
        success_fn_key=values['success_fn_key'],
        input_key=values['input_key']
    )


def create_blackboard(blackboard_data: Dict[str, Any]) -> Blackboard:
    """Create a Blackboard instance from JSON data"""
    blackboard = Blackboard()
    
    # Add each key-value pair to the blackboard
    for key, value in blackboard_data.items():
        blackboard[key] = value
        
    return blackboard


def create_behavior_tree(
    tree_data: Dict[str, Any],
    configs: Dict[str, Dict[str, Any]],
    blackboards: Dict[str, Dict[str, Any]],
    config_objects: Dict[str, Dict[str, Any]]
) -> BehaviorTree:
    """Create a behavior tree from JSON data"""
    # Verify this is a root node
    if tree_data.get('category') != 'root' or tree_data.get('type') != 'Root':
        raise ValueError("Tree must start with a Root node")
    
    # Get tree name and blackboard
    tree_name = tree_data.get('custom_name', '')
    blackboard_id = tree_data.get('blackboard')
    
    # Create empty behavior tree first
    tree = BehaviorTree(tree_name)
    
    # Get the first actual node (child of root)
    if not tree_data.get('children'):
        raise ValueError("Root node must have a child")
    first_node_data = tree_data['children'][0]
    
    # Create the tree starting from the first actual node
    root_node = create_node_from_json(first_node_data, configs, blackboards, config_objects)
    
    # Now set the root node
    tree.set_root(root_node)
    
    # Set blackboard if specified
    if blackboard_id:
        if blackboard_id not in blackboards:
            raise ValueError(f"Blackboard {blackboard_id} not found")
        blackboard_data = blackboards[blackboard_id]
        for key, value in blackboard_data.items():
            tree.blackboard[key] = value
    
    return tree


def create_node_from_json(
    node_data: Dict[str, Any], 
    configs: Dict[str, Dict[str, Any]], 
    blackboards: Dict[str, Dict[str, Any]],
    config_objects: Dict[str, Dict[str, Any]]
) -> TreeNode:
    """Create a node and its children from JSON data"""
    # Get basic node info
    node_type = node_data['type']
    custom_name = node_data.get('custom_name', '')
    
    # Handle Custom nodes first
    if node_type in ['CustomAction', 'CustomCondition']:
        if 'custom_type' not in node_data:
            raise ValueError(f"{node_type} requires custom_type")
        actual_type = node_data['custom_type']
        if actual_type not in node_registry:
            raise ValueError(f"Custom type {actual_type} not found in registry")
        node_class = node_registry[actual_type]
        node = node_class(custom_name)
    else:
        # For all other nodes, verify type exists in registry
        if node_type not in node_registry:
            raise ValueError(f"Node type {node_type} not found in registry")
        
        # Get node class from registry
        node_class = node_registry[node_type]

        # Create the node based on type
        if node_type in ['GenerateAction', 'LogLikelihoodAction', 'LogLikelihoodRollingAction']:
            if 'configs' not in node_data:
                raise ValueError(f"{node_type} requires HFLMConfig and LMActionConfig")
            
            config_ids = node_data['configs']
            if 'HFLMConfig' not in config_ids or 'LMActionConfig' not in config_ids:
                raise ValueError(f"{node_type} requires both HFLMConfig and LMActionConfig")
                
            model_config = config_objects['HFLMConfig'][config_ids['HFLMConfig']]
            action_config = config_objects['LMActionConfig'][config_ids['LMActionConfig']]
            
            node = node_class(model_config, action_config)
            node_data.pop('configs', None)
        else:
            # Create standard node instance
            node = node_class(custom_name)
    
    # Handle configs if present
    if 'configs' in node_data:
        node_configs = {}
        for config_type, config_id in node_data['configs'].items():
            if config_id not in config_objects.get(config_type, {}):
                raise ValueError(f"Config {config_id} of type {config_type} not found")
            node_configs[config_type] = config_objects[config_type][config_id]
        node.configure(node_configs)

    # Handle blackboard if present
    if 'blackboard' in node_data:
        blackboard_id = node_data['blackboard']
        if blackboard_id not in blackboards:
            raise ValueError(f"Blackboard {blackboard_id} not found")
        blackboard_data = blackboards[blackboard_id]
        for key, value in blackboard_data.items():
            node.blackboard[key] = value

    # Recursively create children
    if 'children' in node_data:
        children = node_data['children']
        for child_data in children:
            child = create_node_from_json(child_data, configs, blackboards, config_objects)
            if not isinstance(child, TreeNode):
                continue
            node.add_child(child)

    return node


def load_json_file(file_path: str) -> tuple[Dict[str, BehaviorTree], Dict[str, Dict[str, Any]], Dict[str, Blackboard]]:
    """
    Load behavior trees and associated data from a JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        tuple containing:
            - Dict[str, BehaviorTree]: Dictionary mapping tree names to BehaviorTree objects
            - Dict[str, Dict[str, Any]]: Dictionary of all config objects by type and id
            - Dict[str, Blackboard]: Dictionary of all blackboards by id
    """
    # Read and parse the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the configurations and blackboards from JSON
    config_data = data.get('configs', {})
    blackboard_data = data.get('blackboards', {})
    
    # Create config objects
    config_objects = {
        'HFLMConfig': {},
        'LMActionConfig': {},
        'LMCompletionConfig': {}
    }
    
    for config_id, config_data in config_data.items():
        config_type = config_data['type']
        if config_type == 'HFLMConfig':
            config_objects['HFLMConfig'][config_id] = create_hflm_config(config_data)
        elif config_type == 'LMActionConfig':
            config_objects['LMActionConfig'][config_id] = create_lm_action_config(config_data)
        elif config_type == 'LMCompletionConfig':
            config_objects['LMCompletionConfig'][config_id] = create_lm_completion_config(config_data)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    # Create blackboard objects
    blackboards = {
        board_id: create_blackboard(board_data)
        for board_id, board_data in blackboard_data.items()
    }
    
    # Create all trees
    trees = {}
    for tree_name, tree_data in data.get('trees', {}).items():
        trees[tree_name] = create_behavior_tree(tree_data, config_data, blackboard_data, config_objects)
    
    return trees, config_objects, blackboards