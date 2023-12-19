from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus

from dataclasses import dataclass, field

import torch
from transformers import pipeline

from typing import Optional

@dataclass
class SimplePipelineActionConfig:
    task_name : str
    model : Optional[str] = field(
        default = None
    )
    input_key : Optional[str] = field(
        default = "in"
    )
    output_key : Optional[str] = field(
        default = "out"
    )
    device : Optional[str] = field(
        default = "auto"
    )


class SimplePipelineActionNode(ActionNode):
    def __init__(self, name : str, cfg : SimplePipelineActionConfig):
        super().__init__(name)

        self.task_name = cfg.task_name

        self.input_key = cfg.input_key
        self.output_key = cfg.output_key

        self.device = cfg.device

        if cfg.model:
            self.pipeline = pipeline(cfg.task_name, model=cfg.model, device_map=self.device)
        else:
            self.pipeline = pipeline(cfg.task_name, device_map=self.device)


    def tick(self):
        try:
            input_text = self.blackboard_get(self.input_key)
            output = self.pipeline(input_text)
            self.blackboard_set(self.output_key, output)
            return NodeStatus.SUCCESS            
        except:
            return NodeStatus.FAILURE
