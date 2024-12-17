from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus
from dendron.configs.hflm_action_config import HFLMActionConfig

from typing import Callable, List, Tuple

import types
from hflm.huggingface_model import HFLM
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
    def __init__(self, name: str, cfg: HFLMActionConfig) -> None:
        super().__init__(name)

        self.prompt_key = cfg.input_key
        self.completions_key = cfg.completions_key  # New key for list of completions
        self.output_key = cfg.output_key
        self.device = cfg.device
        self.torch_dtype = cfg.dtype

        self.model = HFLM(
            model=cfg.model, 
            device=self.device, 
            parallelize=cfg.parallelize,
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit
        )

        self.input_processor = None
        self.output_processor = None

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
            log_probs = self.model.loglikelihood(prompt_completion_pairs, disable_tqdm=True)

            if self.output_processor:
                log_probs = self.output_processor(log_probs)

            self.blackboard[self.output_key] = log_probs

            return NodeStatus.SUCCESS
        except Exception as ex:
            print(f"Exception in node {self.name}:")
            print(traceback.format_exc())

            return NodeStatus.FAILURE