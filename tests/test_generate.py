from dendron import NodeStatus, Blackboard
from dendron.actions.generate import GenerateAction
from dendron.actions.configs.hflm_config import HFLMActionConfig
import pytest

def test_generate_action_phi():
    # Create test config for Phi model
    config = HFLMActionConfig(
        model="microsoft/Phi-3-mini-4k-instruct",  # Using phi-2 as it's smaller than 3.5
        input_key="in",
        output_key="out",
        max_new_tokens=100,
        temperature=0.0,  # Deterministic output
        device="cpu",
        parallelize=False
    )
    
    # Set up blackboard with test prompt
    bb = Blackboard()
    bb["in"] = "What is 2+2? Answer with just the number."
    
    # Create and configure node
    node = GenerateAction("test_generate", config)
    node.set_blackboard(bb)
    
    # Execute node
    result = node.execute_tick()

    # Verify results
    assert result == NodeStatus.SUCCESS
    assert "4" in bb["out"]  # Basic sanity check of output

def test_generate_action_with_processors():
    config = HFLMActionConfig(
        model="microsoft/Phi-3-mini-4k-instruct",
        input_key="in",
        output_key="out",
        max_new_tokens=100,
        temperature=0.0,
        device="cpu",
        parallelize=False
    )
    
    def input_processor(self, text):
        return f"Q: {text}\nA:"
        
    def output_processor(self, text):
        return text.split("A:")[-1].strip()
    
    bb = Blackboard()
    bb["in"] = "What is 2+2?"
    
    node = GenerateAction("test_generate", config)
    node.set_blackboard(bb)
    node.set_input_processor(input_processor)
    node.set_output_processor(output_processor)
    
    result = node.execute_tick()
    
    assert result == NodeStatus.SUCCESS
    assert "4" in bb["out"]



if __name__ == "__main__":
    test_generate_action_phi()
    test_generate_action_with_processors()
