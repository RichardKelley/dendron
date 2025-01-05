import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class LMCompletionConfig:
    """
    Configuration for CompletionConditionNode that controls Hugging Face model settings
    and blackboard interactions for completion evaluation.

    Args:
        node_name (str):
            The name of the node that will use this configuration.

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
    """
    _node_name: str
    _completions_key: Optional[str] = field(default="completions_in")
    _logprobs_out_key: Optional[str] = field(default="probs_out")
    _success_fn_key: Optional[str] = field(default="success_fn")
    _input_key: Optional[str] = field(default="in")

    def __init__(
        self,
        node_name: str,
        completions_key: Optional[str] = "completions_in",
        logprobs_out_key: Optional[str] = "probs_out",
        success_fn_key: Optional[str] = "success_fn",
        input_key: Optional[str] = "in",
    ):
        self._node_name = node_name
        self._completions_key = completions_key
        self._logprobs_out_key = logprobs_out_key
        self._success_fn_key = success_fn_key
        self._input_key = input_key
    
    @property
    def node_name(self):
        return self._node_name
    
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

    def to_dict(self) -> dict:
        return {
            "node_name": self.node_name,
            "completions_key": self.completions_key,
            "logprobs_out_key": self.logprobs_out_key,
            "success_fn_key": self.success_fn_key,
            "input_key": self.input_key,
        }