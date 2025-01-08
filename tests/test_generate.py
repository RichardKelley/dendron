from dendron import NodeStatus, Blackboard, BehaviorTree
from dendron.actions.generate_action import GenerateAction
from dendron.configs.hflm_config import HFLMConfig
from dendron.configs.lm_action_config import LMActionConfig

def test_generate_action_phi():
    # Create test config for Phi model

    model_config = HFLMConfig(
        model="microsoft/phi-4",
        device="cuda",
        parallelize=False,
        load_in_8bit=True
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


    print("OUT: ",tree.blackboard)

    # Verify results
    assert result == NodeStatus.SUCCESS
    assert "4" in tree.blackboard["out"]  # Basic sanity check of output

def test_generate_action_with_processors():
    model_config = HFLMConfig(
        model="microsoft/phi-4",
        device="cuda",
        parallelize=False,
        load_in_8bit=True
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
    tree.blackboard["in"] = "What is 2+2?"
    
    result = tree.tick_once()
    

    print(tree.blackboard["out"])
    assert result == NodeStatus.SUCCESS
    assert "4" in tree.blackboard["out"]



if __name__ == "__main__":
    test_generate_action_phi()
    test_generate_action_with_processors()
