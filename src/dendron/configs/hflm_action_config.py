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
    _model: Union[str, PreTrainedModel]
    _tokenizer: Optional[Union[str, PreTrainedTokenizer]] = field(default=None)
    _input_key: Optional[str] = field(default="in")
    _output_key: Optional[str] = field(default="out")
    _max_new_tokens: Optional[int] = field(default=16)
    _completions_key: Optional[str] = field(default="completions")
    _temperature: Optional[float] = field(default=0.0)
    _truncation: Optional[bool] = field(default=False)
    _device: Optional[str] = field(default="cuda")
    _parallelize: Optional[bool] = field(default=False)
    _dtype: Optional[Union[str, torch.dtype]] = field(default="auto")
    _load_in_8bit: Optional[bool] = field(default=False)
    _load_in_4bit: Optional[bool] = field(default=False)
    _add_bos_token: Optional[bool] = field(default=False)
    _max_length: Optional[int] = field(default=None)
    _prefix_token_id: Optional[int] = field(default=None)
    _batch_size: Optional[Union[int, str]] = field(default=-1)
    _max_batch_size: Optional[int] = field(default=1024)
    _offload_folder: Optional[Union[str, os.PathLike]] = field(default="./offload")

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        input_key: Optional[str] = "in",
        output_key: Optional[str] = "out",
        completions_key: Optional[str] = "completions",
        max_new_tokens: Optional[int] = 16,
        temperature: Optional[float] = 0.0,
        truncation: Optional[bool] = False,
        device: Optional[str] = "cuda",
        parallelize: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        add_bos_token: Optional[bool] = False,
        max_length: Optional[int] = None,
        prefix_token_id: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = -1,
        max_batch_size: Optional[int] = 1024,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload"
    ):
        # Map the public parameter names to internal underscore names
        self._model = model
        self._tokenizer = tokenizer
        self._input_key = input_key
        self._output_key = output_key
        self._completions_key = completions_key
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._truncation = truncation
        self._device = device
        self._parallelize = parallelize
        self._dtype = dtype
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._add_bos_token = add_bos_token
        self._max_length = max_length
        self._prefix_token_id = prefix_token_id
        self._batch_size = batch_size
        self._max_batch_size = max_batch_size
        self._offload_folder = offload_folder

    @property
    def model_name(self):
        if isinstance(self._model, str):
            return self._model 
        elif isinstance(self._model, PreTrainedModel):
            return self._model.name_or_path
        
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def input_key(self):
        return self._input_key
    
    @property
    def output_key(self):
        return self._output_key
    
    @property
    def completions_key(self):
        return self._completions_key
    
    @property
    def max_new_tokens(self):
        return self._max_new_tokens
    
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def truncation(self):
        return self._truncation
    
    @property
    def device(self):
        return self._device
    
    @property
    def parallelize(self):
        return self._parallelize
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def load_in_8bit(self):
        return self._load_in_8bit
    
    @property
    def load_in_4bit(self):
        return self._load_in_4bit
    
    @property
    def add_bos_token(self):
        return self._add_bos_token
    
    @property
    def max_length(self):
        return self._max_length
    
    @property
    def prefix_token_id(self):
        return self._prefix_token_id
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def max_batch_size(self):
        return self._max_batch_size
    
    @property
    def offload_folder(self):
        return self._offload_folder
