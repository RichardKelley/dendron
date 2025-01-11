from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus
from dendron.configs.lm_action_config import LMActionConfig
from dendron.configs.hflm_config import HFLMConfig
from dendron.behavior_tree import BehaviorTree

from typing import Callable

import types
import traceback

class LogLikelihoodAction(ActionNode):
    """
    An action node that uses a language model to compute log-likelihoods
    for a list of completion strings given a prompt.

    This node is based on the HFLM library, and will
    download the model that you specify by name. This can take a long 
    time and/or use a lot of storage, depending on the model you name.

    Args:
        name (str):
            The given name of this node.
        cfg (HFLMActionConfig):
            The configuration object for this model.
    """
    def __init__(self, model_cfg: HFLMConfig, node_cfg: LMActionConfig) -> None:
        super().__init__(node_cfg.node_name)

        self.prompt_key = node_cfg.input_key
        self.completions_key = node_cfg.completions_key 
        self.output_key = node_cfg.output_key

        self.input_processor = None
        self.output_processor = None
        self.node_config = node_cfg
        self.model_config = model_cfg

    def set_model(self, new_model) -> None:
        """
        Set a new model instance.
        """
        self.model = new_model

    def set_input_processor(self, f: Callable) -> None:
        """
        Set the input processor to use during `tick()`s. 

        An input processor is applied to the prompt text and completion list
        stored in the blackboard, and can be used to preprocess them. The 
        processor function should be a map from (str, List[str]) to (str, List[str]).

        Args:
            f (Callable):
                The input processor function to use. Should be a callable
                object that maps (self, Any) to (str, List[str]).
        """
        self.input_processor = types.MethodType(f, self)

    def set_output_processor(self, f: Callable) -> None:
        """
        Set the output processor to use during `tick()`s.

        An output processor is applied to the log-likelihoods returned by the model,
        before they are written to the output slot of the blackboard.
        The function should be a map from List[Tuple[float, bool]] to Any.

        Args:
            f (Callable):
                The output processor function. Should be a callable object
                that maps from (self, List[Tuple[float, bool]]) to Any.
        """
        self.output_processor = types.MethodType(f, self)

    def tick(self) -> NodeStatus:
        """
        Execute a tick, consisting of the following steps:

        - Retrieve a prompt and list of completions from the node's blackboard
        - Apply the input processor, if one exists
        - Compute log-likelihoods for each completion given the prompt
        - Apply the output processor, if one exists
        - Write the result back to the blackboard

        Returns SUCCESS if everything works, FAILURE if there's an exception.
        """
        try:
            prompt = self.blackboard[self.prompt_key]
            completions = self.blackboard[self.completions_key]

            if self.input_processor:
                prompt, completions = self.input_processor(prompt, completions)
            
            # Create iterator of (prompt, completion) pairs
            prompt_completion_pairs = ((prompt, completion) for completion in completions)
            
            # Compute log-likelihoods
            log_probs = self.tree.get_model(self.model_config.model_name).loglikelihood(prompt_completion_pairs, disable_tqdm=True)

            if self.output_processor:
                log_probs = self.output_processor(log_probs)

            self.blackboard[self.output_key] = log_probs

            return NodeStatus.SUCCESS
        except Exception as ex:
            print(f"Exception in node {self.name}:")
            print(traceback.format_exc())

            return NodeStatus.FAILURE

    def set_tree(self, tree : BehaviorTree) -> None:
        """
        Set the behavior tree for this node, which includes setting up the blackboard
        and registering the model configuration with the tree.

        Args:
            tree (BehaviorTree):
                The behavior tree this node belongs to.
        """
        self.tree = tree
        self.set_blackboard(tree.blackboard)
        tree.add_model(self.model_config)