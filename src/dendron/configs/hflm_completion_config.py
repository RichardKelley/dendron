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
    _model: Union[str, PreTrainedModel]
    _completions_key: Optional[str] = field(default="completions_in")
    _logprobs_out_key: Optional[str] = field(default="probs_out")
    _success_fn_key: Optional[str] = field(default="success_fn")
    _input_key: Optional[str] = field(default="in")
    _device: Optional[str] = field(default="cuda")
    _parallelize: Optional[bool] = field(default=False)
    _dtype: Optional[Union[str, torch.dtype]] = field(default="auto")
    _load_in_8bit: Optional[bool] = field(default=False)
    _load_in_4bit: Optional[bool] = field(default=False)
    _add_bos_token: Optional[bool] = field(default=False)
    _offload_folder: Optional[Union[str, os.PathLike]] = field(default="./offload")

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        completions_key: Optional[str] = "completions_in",
        logprobs_out_key: Optional[str] = "probs_out",
        success_fn_key: Optional[str] = "success_fn",
        input_key: Optional[str] = "in",
        device: Optional[str] = "cuda",
        parallelize: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        add_bos_token: Optional[bool] = False,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload"
    ):
        # Map the public parameter names to internal underscore names
        self._model = model
        self._completions_key = completions_key
        self._logprobs_out_key = logprobs_out_key
        self._success_fn_key = success_fn_key
        self._input_key = input_key
        self._device = device
        self._parallelize = parallelize
        self._dtype = dtype
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._add_bos_token = add_bos_token
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
    def completions_key(self):
        return self._completions_key
    
    @property
    def logprobs_out_key(self):
        return self._logprobs_out_key
    
    @property
    def success_fn_key(self):
        return self._success_fn_key
    
    @property
    def input_key(self):
        return self._input_key
    
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
    def offload_folder(self):
        return self._offload_folder
