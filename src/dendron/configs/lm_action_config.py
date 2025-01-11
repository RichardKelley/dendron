import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class LMActionConfig:
    """
    Configuration for LM action nodes (generate_action, loglikelihood_action, and 
    loglikelihood_rolling_action).

    Controls LM generation and output processes as well as blackboard interactions.

    Args:

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
    """
    _node_name: str
    _input_key: Optional[str] = field(default="in")
    _output_key: Optional[str] = field(default="out")
    _max_new_tokens: Optional[int] = field(default=16)
    _completions_key: Optional[str] = field(default="completions")
    _temperature: Optional[float] = field(default=0.0)
    _truncation: Optional[bool] = field(default=False)
    _max_length: Optional[int] = field(default=None)
    _prefix_token_id: Optional[int] = field(default=None)
    _batch_size: Optional[Union[int, str]] = field(default=-1)
    _max_batch_size: Optional[int] = field(default=1024)

    def __init__(
        self,
        node_name: str,
        input_key: Optional[str] = "in",
        output_key: Optional[str] = "out",
        completions_key: Optional[str] = "completions",
        max_new_tokens: Optional[int] = 16,
        temperature: Optional[float] = 0.0,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        prefix_token_id: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = -1,
        max_batch_size: Optional[int] = 1024,
    ):
        self._node_name = node_name
        self._input_key = input_key
        self._output_key = output_key
        self._completions_key = completions_key
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._truncation = truncation
        self._max_length = max_length
        self._prefix_token_id = prefix_token_id
        self._batch_size = batch_size
        self._max_batch_size = max_batch_size

    @property
    def node_name(self):
        return self._node_name
    
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
    
    def to_dict(self) -> dict:
        return {
            "node_name": self.node_name,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "completions_key": self.completions_key,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "prefix_token_id": self.prefix_token_id,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
        }