import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class HFLMCompletionConfig:
    """
    Configuration for CompletionConditionNode that controls Hugging Face model settings
    and blackboard interactions for completion evaluation.

    Args:
        model (Union[str, PreTrainedModel]):
            Either a string specifying the Hugging Face model name (including organization name),
            or a pre-loaded PreTrainedModel instance.

        completions_key (Optional[str]):
            The blackboard key for reading and writing completions to evaluate.
            The value should be a list of strings, each representing one completion.
            Defaults to "completions_in".

        logprobs_out_key (Optional[str]):
            The blackboard key where the output log probabilities dictionary will be written.
            Defaults to "probs_out".

        success_fn_key (Optional[str]):
            The blackboard key for the success predicate function. The predicate should
            accept a completion string and return a NodeStatus.
            Defaults to "success_fn".

        input_key (Optional[str]):
            The blackboard key used to read the input text that the model will process.
            Defaults to "in".

        device (Optional[str]):
            The device where the model will be loaded and executed.
            Defaults to "cuda".

        parallelize (Optional[bool]):
            Whether to parallelize the model across multiple GPUs using model parallelism.
            Defaults to False.

        dtype (Optional[Union[str, torch.dtype]]):
            Data type for model weights and computations. Can be "auto", "float16", "bfloat16", 
            or a torch.dtype. "auto" will attempt to choose the best dtype based on the device.
            Defaults to "auto".

        load_in_8bit (Optional[bool]):
            Whether to load the model in 8-bit mode. Defaults to False.

        load_in_4bit (Optional[bool]):
            Whether to load the model in 4-bit mode. Defaults to False.

        add_bos_token (Optional[bool]):
            Whether to add a beginning-of-sequence token to the input. Required for some
            models like Gemma. Defaults to False.

        offload_folder (Optional[Union[str, os.PathLike]]):
            Directory path where model weights will be offloaded when using disk 
            offloading for large models. Defaults to "./offload".
    """
    model: Union[str, PreTrainedModel]
    completions_key: Optional[str] = field(default="completions_in")
    logprobs_out_key: Optional[str] = field(default="probs_out")
    success_fn_key: Optional[str] = field(default="success_fn")
    input_key: Optional[str] = field(default="in")
    device: Optional[str] = field(default="cuda")
    parallelize: Optional[bool] = field(default=False)
    dtype: Optional[Union[str, torch.dtype]] = field(default="auto")
    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    add_bos_token: Optional[bool] = field(default=False)
    offload_folder: Optional[Union[str, os.PathLike]] = field(default="./offload")