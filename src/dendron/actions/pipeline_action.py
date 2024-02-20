from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus

from dataclasses import dataclass, field

import torch
from transformers import pipeline

from typing import Optional

@dataclass
class PipelineActionConfig:
    """
    Configuration for a PipelineAction.

    The options in this object control what Hugging Face task and model
    are used and how the node interacts with the blackboard.

    Args:
        task_name (`str`):
            The name of the Hugging Facetask to use. This should be a
            valid HF task name. For an overview of the tasks that HF
            supports, see https://huggingface.co/tasks.
        model (`Optional[str]`):
            Optional name of a model to use. This should be a valid
            name corresponding to a Hugging Face model name (including
            the user name). Defaults to None, in which case the default
            model for the pipeline task will be used.
        input_key (`Optional[str]`):
            The blackboard key to use for writing and reading the prompt that 
            this node will consume. Defaults to "in".
        output_key (`Optional[str]`):
            The blackboard key to use for writing and reading the text generated
            by this node. Defaults to "out".
        device (`Optional[str]`):
            The device that should be used with the model. Examples include
            "cpu", "cuda", and "auto". Defaults to "auto".
    """
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


class PipelineAction(ActionNode):
    """
    An action node that uses a Hugging Face transformers pipeline object
    to execute a behavior. This enables easy access to functionality such
    as sentiment classification that is wrapped in a Pipeline. This is 
    also useful for quick prototyping with HF defaults.

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
        cfg (`PipelineActionConfig`):
            The configuration object for this model.
    """
    def __init__(self, name : str, cfg : PipelineActionConfig) -> None:
        super().__init__(name)

        self.task_name = cfg.task_name

        self.input_key = cfg.input_key
        self.output_key = cfg.output_key

        self.device = cfg.device

        if cfg.model:
            self.pipeline = pipeline(cfg.task_name, model=cfg.model, device_map=self.device)
        else:
            self.pipeline = pipeline(cfg.task_name, device_map=self.device)


    def tick(self) -> NodeStatus:
        """
        Execute a tick, consisting of the following steps:

        - Retrieve a prompt from the node's blackboard.
        - Apply the pipeline object to the input text.
        - Write the output to the blackboard.

        If any of the above fail, then the node returns a status of
        `FAILURE`. Otherwise the node returns a status of `SUCCESS`.
        """
        try:
            input_text = self.blackboard[self.input_key]
            output = self.pipeline(input_text)
            self.blackboard[self.output_key] = output
            return NodeStatus.SUCCESS            
        except:
            return NodeStatus.FAILURE
