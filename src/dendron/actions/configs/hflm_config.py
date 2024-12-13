import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class HFLMActionConfig:
    """
    Configuration for a HFLMAction.
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
        defualt = "cuda"
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
