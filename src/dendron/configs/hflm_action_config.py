import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class HFLMActionConfig:
    """
    Configuration for all of the HFLM action nodes (generate_action, loglikelihood_action, and loglikelihood_rolling_action).

    The options in this object control what Hugging Face model is used
    and how the node interacts with the blackboard for the completion condition.

    Args:
        model (`str`):
            The name of the model to use. This should be a valid name
            corresponding to a Hugging Face model name (including the user
            name).
        input_key (`Optional[str]`):
            The blackboard key to use for writing and reading the prefix that 
            this node will consume. Defaults to "in".
        output_key (`Optional[str]`):
            The blackboard key to use for writing the suffix that 
            this node will produce. Defaults to "output". (TODO: check this phrasing against Richard's version)
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
    offload_folder : Optional[Union[str, os.PathLike]] = field(
        default = "./offload"
    )
