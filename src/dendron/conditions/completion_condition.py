from ..condition_node import ConditionNode
from ..basic_types import NodeStatus, Quantization

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from typing import Optional, Callable, List

import traceback

@dataclass
class CompletionConditionConfig:
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
        device (`Optional[str]`):
            The device that should be used with the model. Examples include
            "cpu", "cuda", and "auto". Defaults to "auto".
        load_in_8bit (`Optional[bool]`):
            Optional boolean indicating whether or not to use eight-bit quantization
            from bitsandbytes. When available, will typically decrease memory usage
            and increase inference speed. Defaults to `False`.
        load_in_4bit (`Optional[bool]`):
            Optional boolean indicating whether or not to use four-bit quantization
            from bitsandbytes. When available, will typically decrease memory usage
            and increase inference speed. If you observe degraded performance, try
            eight-bit quanitization instead. Defaults to `False`.
        torch_dtype (`torch.dtype`):
            The dtype to use for torch tensors. Defaults to `torch.float16`. You may
            need to change this depending on your quantization choices.
        use_flash_attn_2 (`Optional[bool]`):
            Optional bool controlling whether or not to use Flash Attention 2. Defaults
            to `False` in case you haven't installed flash attention. Substantially
            speeds up inference. 
    """
    model_name : str
    completions_key : Optional[str] = field(
        default = "completions_in"
    )
    logprobs_out_key : Optional[str] = field (
        default = "probs_out"
    )
    success_fn_key : Optional[str] = field(
        default = "success_fn"
    )
    auto_load : Optional[bool] = field(
        default = True
    )
    input_key : Optional[str] = field(
        default = "in"
    )
    device : Optional[str] = field(
        default = "auto"
    )
    load_in_8bit : Optional[bool] = field(
        default = False
    )
    load_in_4bit : Optional[bool] = field(
        default = False
    )
    torch_dtype : Optional[torch.dtype] = field(
        default = torch.float16
    )
    use_flash_attn_2 : Optional[bool] = field(
        default = False
    )

class CompletionCondition(ConditionNode):
    """
    A completion condition node uses a causal language model to evaluate
    the relative likelihood of several different completions of a prompt,
    returning `SUCCESS` or `FAILURE` using a user-provided function that
    selects a status based on the most likely completion.

    This node tends to run quickly and gives useful answers, but if you
    use this node you should be aware of the perils of "surface form
    competition", documented in the paper by Holtzman et al. (see 
    https://arxiv.org/abs/2104.08315).

    This node is based on the Hugging Face transformers library, and will
    download the model that you specify by name. This can take a long 
    time and/or use a lot of storage, depending on the model you name.

    There are enough configuration options for this type of node that
    the options have all been placed in a dataclass config object. See 
    the documentation for that object to learn about the many options
    available to you.

    Args:
        name (`str`):
            The given name of this node.
        cfg (`CompletionConditionNodeConfig`):
            The configuration object for this model.
    """
    def __init__(self, name : str, cfg : CompletionConditionConfig) -> None:
        super().__init__(name)
        self.input_key = cfg.input_key
        self.device = cfg.device

        self.torch_dtype = cfg.torch_dtype
        self.completions_key = cfg.completions_key
        self.success_fn_key = cfg.success_fn_key

        self.logprobs_out_key = cfg.logprobs_out_key

        self.bnb_cfg = BitsAndBytesConfig()
        
        match cfg.load_in_4bit, cfg.load_in_8bit:
            case True, True:
                self.bnb_cfg.load_in_4bit = True
                self.bnb_cfg.bnb_4bit_compute_dtype = cfg.torch_dtype
            case True, False:
                self.bnb_cfg.load_in_4bit = True
                self.bnb_cfg.bnb_4bit_compute_dtype = cfg.torch_dtype
            case False, True:
                self.bnb_cfg.load_in_8bit = True
            case False, False:
                pass
                
        if cfg.use_flash_attn_2:
            self.attn_implementation = "flash_attention_2"
        else:
            self.attn_implementation = "sdpa"

        if cfg.auto_load:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                low_cpu_mem_usage=True,
                attn_implementation=self.attn_implementation,
                quantization_config=self.bnb_cfg
            )
                    
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.model = None
            self.tokenizer = None
            self.completions = []

    def set_model(self, new_model) -> None:
        """
        Set a new model to use for generating text.
        """
        self.model = new_model
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(new_model.name_or_path)

    def tick(self) -> NodeStatus:
        """
        Execute a tick, consisting of the following steps:

        - Retrieve the input prefix from the blackboard.
        - Retrieve the list of completion options from the blackboard.
        - Retrieve the success predicate from the blackboard.
        - Tokenize all of the possible completions, padding as needed.
        - Evaluate the model on the tokenized batch of completions.
        - Compute the "log probabilities" of each completion.
        - Apply the success predicate to the completion with the highest
          log probability.
        - Return the status computed by the success predicate.

        If any of the above fail, the exception text is printed and the node
        returns a status of `FAILURE`. Otherwise the node returns `SUCCESS`.
        """
        try:
            input_prefix = self.blackboard[self.input_key]
            completions = self.blackboard[self.completions_key]
            success_fn = self.blackboard[self.success_fn_key]

            log_probs = np.zeros(len(completions))
            texts = [input_prefix + s for s in completions]

            # Based on discussion/code at: https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
            input_ids = self.tokenizer(texts, padding=True, return_tensors="pt").input_ids
            outputs = self.model(input_ids)
            probs = torch.log_softmax(outputs.logits, dim=-1).detach()

            probs = probs[:, :-1, :]
            input_ids = input_ids[:, 1:]
            gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

            for i, (input_sentence, input_probs) in enumerate(zip(input_ids, gen_probs)):
                for token, p in zip(input_sentence, input_probs):
                    if token not in self.tokenizer.all_special_ids:
                        log_probs[i] += p.item()
            
            self.blackboard[self.logprobs_out_key] = {completions[i] : log_probs[i] for i in range(len(log_probs))}

            best_completion = completions[log_probs.argmax()]

            return success_fn(best_completion)

        except Exception as ex:
            print(f"Exception in node {self.name}:")
            print(traceback.format_exc())
            return NodeStatus.FAILURE