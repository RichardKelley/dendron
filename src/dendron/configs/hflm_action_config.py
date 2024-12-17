import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class HFLMActionConfig:
    """
    Configuration for HFLM action nodes (generate_action, loglikelihood_action, and loglikelihood_rolling_action).
    Controls the Hugging Face model configuration and blackboard interactions.

    Args:
        model (Union[str, PreTrainedModel]):
            Either a string specifying the Hugging Face model name (including organization name),
            or a pre-loaded PreTrainedModel instance.
        
        tokenizer (Optional[Union[str, PreTrainedTokenizer]]):
            The tokenizer to use with the model. Can be either a string specifying the 
            tokenizer name or a pre-loaded PreTrainedTokenizer instance. If None, 
            will attempt to load the default tokenizer for the model.

        input_key (Optional[str]):
            The blackboard key used to read the input text that the model will process.
            Defaults to "in".

        output_key (Optional[str]):
            The blackboard key where the model's output will be written.
            Defaults to "out".

        max_new_tokens (Optional[int]):
            Maximum number of tokens to generate in one forward pass.
            Defaults to 16.

        temperature (Optional[float]):
            Sampling temperature for generation. Higher values (e.g., 0.8) make the output
            more random, while lower values (e.g., 0.2) make it more deterministic.
            Defaults to 0.0 (deterministic).

        truncation (Optional[bool]):
            Whether to truncate input sequences that exceed the model's maximum length.
            Defaults to False.

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

        max_length (Optional[int]):
            Maximum allowed length for the combined input and output sequence.
            If None, will use the model's default max length. Defaults to None.

        prefix_token_id (Optional[int]):
            Token ID to be prepended to all inputs. If specified, this token will be
            added before the input sequence. Defaults to None.

        batch_size (Optional[Union[int, str]]):
            Batch size for processing multiple sequences. Use -1 for automatic batching.
            Defaults to -1.

        max_batch_size (Optional[int]):
            Maximum allowed batch size when using automatic batching.
            Defaults to 1024.

        offload_folder (Optional[Union[str, os.PathLike]]):
            Directory path where model weights will be offloaded when using disk 
            offloading for large models. Defaults to "./offload".
    """
    model: Union[str, PreTrainedModel]
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = field(default=None)
    input_key: Optional[str] = field(default="in")
    output_key: Optional[str] = field(default="out")
    max_new_tokens: Optional[int] = field(default=16)
    temperature: Optional[float] = field(default=0.0)
    truncation: Optional[bool] = field(default=False)
    device: Optional[str] = field(default="cuda")
    parallelize: Optional[bool] = field(default=False)
    dtype: Optional[Union[str, torch.dtype]] = field(default="auto")
    load_in_8bit: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    add_bos_token: Optional[bool] = field(default=False)
    max_length: Optional[int] = field(default=None)
    prefix_token_id: Optional[int] = field(default=None)
    batch_size: Optional[Union[int, str]] = field(default=-1)
    max_batch_size: Optional[int] = field(default=1024)
    offload_folder: Optional[Union[str, os.PathLike]] = field(default="./offload")
