from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus, Quantization

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Optional, Callable

@dataclass
class CausalLMActionConfig:
    model_name : str
    input_key : Optional[str] = field(
        default = "in"
    )
    output_key : Optional[str] = field(
        default = "out"
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
    max_new_tokens : Optional[int] = field(
        default = 16
    )
    do_sample : Optional[bool] = field(
        default = False
    )
    top_p : Optional[float] = field(
        default = 1.0
    )

class CausalLMAction(ActionNode):
    def __init__(self, name : str, cfg : CausalLMActionConfig):
        super().__init__(name)

        self.input_key = cfg.input_key
        self.output_key = cfg.output_key
        self.device = cfg.device

        self.max_new_tokens = cfg.max_new_tokens
        self.do_sample = cfg.do_sample
        self.top_p = cfg.top_p

        match cfg.load_in_4bit, cfg.load_in_8bit:
            case True, True:
                self.quantization = Quantization.FourBit
            case True, False:
                self.quantization = Quantization.FourBit
            case False, True:
                self.quantization = Quantization.EightBit
            case False, False:
                self.quantization = Quantization.NoQuantization

        match self.quantization:
            case Quantization.NoQuantization:
                self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            case Quantization.FourBit:
                self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, load_in_4bit=True, device_map=self.device)
            case Quantization.EightBit:
                self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, load_in_8bit=True, device_map=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.pre_tick_fn = None
        self.post_tick_fn = None

    def set_pre_tick(self, f : Callable[str, str]):
        self.pre_tick_fn = f

    def set_post_tick(self, f : Callable[str, str]):
        self.post_tick_fn = f

    def tick(self):
        try:
            input_text = self.blackboard[self.input_key]

            if self.pre_tick_fn:
                input_text = self.pre_tick_fn(input_text)

            input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, top_p=self.top_p)
            output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if self.post_tick_fn is not None:
                output_text = self.post_tick_fn(self, output_text)

            self.blackboard[self.output_key] = output_text

            return NodeStatus.SUCCESS
        except Exception as ex:
            print(f"Exception ({self.name}): {ex}")
            return NodeStatus.FAILURE