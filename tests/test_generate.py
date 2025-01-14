from dendron import NodeStatus, Blackboard, BehaviorTree
from dendron.actions.generate_action import GenerateAction
from dendron.configs.hflm_config import HFLMConfig
from dendron.configs.lm_action_config import LMActionConfig

from dendron.util import default_device

import pytest
import torch 

@pytest.fixture(autouse=True)
def setup_device():
    # Force device initialization before any tests
    if torch.backends.mps.is_available():
        torch.device("mps")

def test_generate_action_llama():
    # Create test config for llama model

    model_config = HFLMConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        device=default_device(),
        parallelize=False,
        load_in_8bit=False
    )

    node_config = LMActionConfig(
        node_name="GenerateAction",
        input_key="in",
        output_key="out",
        max_new_tokens=100,
        temperature=0.0,
    )
    
    # Create and configure node
    node = GenerateAction(model_config, node_config)
    
    tree = BehaviorTree("generate-tree") 
    tree.set_root(node)

    tree.blackboard["in"] = "What is 2+2?"

    # Execute node
    result = tree.tick_once()

    # Verify results
    assert result == NodeStatus.SUCCESS
    assert "4" in tree.blackboard["out"]  # Basic sanity check of output

def test_generate_action_with_processors():
    model_config = HFLMConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        device=default_device(),
        parallelize=False,
        load_in_8bit=False
    )

    node_config = LMActionConfig(
        node_name="GenerateAction",
        input_key="in",
        output_key="out",
        max_new_tokens=100,
        temperature=0.0,
    )
    
    def input_processor(self, text):
        return f"Q: {text}\nA:"
        
    def output_processor(self, text):
        return text.split("A:")[-1].strip()
        
    node = GenerateAction(model_config, node_config)
    node.set_input_processor(input_processor)
    node.set_output_processor(output_processor)    

    tree = BehaviorTree("generate-tree") 
    tree.set_root(node)
    tree.blackboard["in"] = "What is 2+2? Be sure to preceed the answer with A:"
        
    result = tree.tick_once()
    
    assert result == NodeStatus.SUCCESS
    assert "4" in tree.blackboard["out"]

if __name__ == "__main__":
    test_generate_action_llama()
    test_generate_action_with_processors()
