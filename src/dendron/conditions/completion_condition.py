from ..condition_node import ConditionNode
from ..basic_types import NodeStatus, Quantization
from dendron.configs.hflm_completion_config import HFLMCompletionConfig
import numpy as np
import torch
from hflm.huggingface_model import HFLM
import traceback

argmax = lambda lst: max(enumerate(lst), key=lambda x: x[1])[0]

class CompletionCondition(ConditionNode):
    """
    A completion condition node uses a causal language model to evaluate
    the relative likelihood of several different completions of a prompt,
    returning `SUCCESS` or `FAILURE` using a user-provided function that
    selects a status based on the most likely completion.

    This node tends to run quickly and gives useful answers, but if you
    use this node you should be aware of the perils of "surface form
    competition", documented in the paper by Holtzman et al. (see 
    https://arxiv.org/abs/2104.08315).

    This node is based on the HFLM library, and will
    download the model that you specify by name. This can take a long 
    time and/or use a lot of storage, depending on the model you name.

    There are enough configuration options for this type of node that
    the options have all been placed in a dataclass config object. See 
    the documentation for that object to learn about the many options
    available to you.

    Args:
        name (`str`):
            The given name of this node.
        cfg (`CompletionConditionNodeConfig`):
            The configuration object for this model.
    """
    def __init__(self, name : str, cfg : HFLMCompletionConfig) -> None:
        super().__init__(name)
        self.input_key = cfg.input_key
        self.device = cfg.device
        print(self.device)

        self.completions_key = cfg.completions_key
        self.success_fn_key = cfg.success_fn_key

        self.logprobs_out_key = cfg.logprobs_out_key

        self.model = HFLM(
            model=cfg.model, 
            device=self.device, 
            parallelize=cfg.parallelize,
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit
        )
    def set_model(self, new_model) -> None:
        """
        Set a new model to use for generating text.
        """
        self.model = new_model

    def tick(self) -> NodeStatus:
        """
        Execute a tick, consisting of the following steps:

        - Retrieve the input prefix from the blackboard.
        - Retrieve the list of completion options from the blackboard.
        - Retrieve the success predicate from the blackboard.
        - Compute the log probabilities of each completion.
        - Apply the success predicate to the completion with the highest
          log probability.
        - Return the status computed by the success predicate.

        If any of the above fail, the exception text is printed and the node
        returns a status of `FAILURE`. Otherwise the node returns `SUCCESS`.
        """
        try:
            input_prefix = self.blackboard[self.input_key]
            completions = self.blackboard[self.completions_key]
            success_fn = self.blackboard[self.success_fn_key]

            log_probs = self.model.loglikelihood(((input_prefix, s) for s in completions), disable_tqdm=True)

            self.blackboard[self.logprobs_out_key] = {completions[i] : log_probs[i] for i in range(len(log_probs))}

            best_completion = completions[argmax(log_probs)]

            return success_fn(best_completion)

        except Exception as ex:
            print(f"Exception in node {self.name}:")
            print(traceback.format_exc())
            return NodeStatus.FAILURE