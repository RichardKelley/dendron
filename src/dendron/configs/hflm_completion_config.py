import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class HFLMCompletionConfig:
    """
    Configuration for a CompletionConditionNode.

    The options in this object control what Hugging Face model is used
    and how the node interacts with the blackboard for the completion condition.

    Args:
        model (`str`):
            The name of the model to use. This should be a valid name
            corresponding to a Hugging Face model name (including the user
            name).
        completions_key (`Optional[str]`):
            The blackboard key to read and write the completions to evaluate
            upon a `tick()` call. The value stored here should be a list of
            strings, each string representing one completion. Defaults to
            "completions_in".
        logprobs_out_key (`Optional[str]`):
            The blackboard key to write a dictionary containing the output 
            log probabilities.
        success_fn_key (`Optional[str]`):
            The blackboard key to read and write the success predicate that
            determines the status that is ultimately returned upon a `tick()`
            call. The predicate should accept a completion string as input 
            and return a `NodeStatus`. Defaults to "success_fn".
        input_key (`Optional[str]`):
            The blackboard key to use for writing and reading the prefix that 
            this node will consume. Defaults to "in".
        device (`Optional[str]`):
            The device where the HFLM model will be loaded. Defaults to "cuda".
        parallelize (`Optional[bool]`):
            Boolean variable for whether to parallelize the model across
            multiple GPUs. Defaults to False.
        dtype (`Optional[Union[str, torch.dtype]]`):
            The datatype of the input? Model?
            Defaults to "auto".
            TODO: get clarification on dtype and if we even need it anymore
        add_bos_token (`Optional[bool]`):
            Boolean variable to set the BOS token in the tokenization.
            Needed for Gemma models. Defaults to False.
        offload_folder (`Optional[Union[str, os.PathLike]]`):
            Directory to offload the model to when using disk space
            for parallelization. Defaults to "./offload". 

    """
    model : Union[str, PreTrainedModel]
    completions_key : Optional[str] = field(
        default = "completions_in"
    )
    logprobs_out_key : Optional[str] = field (
        default = "probs_out"
    )
    success_fn_key : Optional[str] = field(
        default = "success_fn"
    )
    input_key : Optional[str] = field(
        default = "in"
    )
    device : Optional[str] = field(
        default = "cuda"
    )
    parallelize : Optional[bool] = field(
        default = False
    )
    dtype : Optional[Union[str, torch.dtype]] = field(
        default = "auto"
    ) # NB: MLX incompatibility
    add_bos_token : Optional[bool] = field( 
        default = False 
    ) # Need for Gemma-2
    offload_folder : Optional[Union[str, os.PathLike]] = field(
        default = "./offload"
    )