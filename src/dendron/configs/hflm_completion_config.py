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
    and how the node interacts with the blackboard.

    Args:
        model_name (`str`):
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
        auto_load (`Optional[bool]`):
            An optional boolean indicating whether or not to automatically 
            load model either from disk or the Hugging Face hub. If `False`,
            the user is responsible for ensuring that a model is loaded
            before the first `tick()` is triggered. Defaults to `True`.
        input_key (`Optional[str]`):
            The blackboard key to use for writing and reading the prefix that 
            this node will consume. Defaults to "in".
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
    tokenizer : Optional[
        Union[
            str, PreTrainedTokenizer
        ]
    ] = field(
        default = None
    )
    input_key : Optional[str] = field(
        default = "in"
    )
    output_key : Optional[str] = field(
        default = "out"
    )
    max_new_tokens : Optional[int] = field(
        default = 16
    )
    temperature : Optional[float] = field(
        default = 0.0
    )
    truncation : Optional[bool] = field(
        default = False
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
    max_length : Optional[int] = field(
        default = None
    )
    prefix_token_id : Optional[int] = field(
        default = None
    )
    batch_size : Optional[Union[int,str]] = field(
        default = -1
    )
    max_batch_size : Optional[int] = field(
        default = 1024
    )
    parallelize : Optional[bool] = field(
        default = False
    )
    max_memory_per_gpu : Optional[Union[int, str]] = field(
        default = None
    )
    max_cpu_memory : Optional[Union[int, str]] = field(
        default = None
    )
    offload_folder : Optional[Union[str, os.PathLike]] = field(
        default = "./offload"
    )