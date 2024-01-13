from ..condition_node import ConditionNode
from ..basic_types import NodeStatus, Quantization

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Optional, Callable, List

@dataclass
class CompletionConditionNodeConfig:
    model_name : str
    completions : List[str] = field(
        default_factory = lambda: []
    )
    success_fn : Callable = field(
        default = lambda x: NodeStatus.FAILURE
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

class CompletionConditionNode(ConditionNode):
    def __init__(self, name : str, cfg : CompletionConditionNodeConfig):
        super().__init__(name)
        self.input_key = cfg.input_key
        self.device = cfg.device

        self.torch_dtype = cfg.torch_dtype
        self.completions = cfg.completions
        self.success_fn = cfg.success_fn

        match cfg.load_in_4bit, cfg.load_in_8bit:
            case True, True:
                self.quantization = Quantization.FourBit
            case True, False:
                self.quantization = Quantization.FourBit
            case False, True:
                self.quantization = Quantization.EightBit
            case False, False:
                self.quantization = Quantization.NoQuantization

        if cfg.use_flash_attn_2:
            self.attn_implementation = "flash_attention_2"
        else:
            self.attn_implementation = "sdpa"

        if cfg.auto_load:
            match self.quantization:
                case Quantization.NoQuantization:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg.model_name,
                        torch_dtype=self.torch_dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation=self.attn_implementation
                    )
                case Quantization.FourBit:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg.model_name,
                        load_in_4bit=True,
                        torch_dtype=self.torch_dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation=self.attn_implementation
                    )
                case Quantization.EightBit:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        cfg.model_name,
                        load_in_8bit=True,
                        torch_dtype=self.torch_dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation=self.attn_implementation
                    )
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.model = None
            self.tokenizer = None
            self.completions = []

    def set_model(self, new_model):
        self.model = new_model
        self.tokenizer = AutoTokenizer.from_pretrained(new_model.name_or_path)

    def tick(self):
        try:
            log_probs = np.zeros(len(self.completions))
            input_prefix = self.blackboard[self.input_key]
            texts = [input_prefix + s for s in self.completions]

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
            
            best_completion = self.completions[log_probs.argmax()]

            return self.success_fn(best_completion)

        except Exception as ex:
            print(f"Exception ({self.name}): {ex}")
            return NodeStatus.FAILURE