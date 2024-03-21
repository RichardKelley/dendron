from dendron.action_node import ActionNode
from dendron.basic_types import NodeStatus, Quantization

from dataclasses import dataclass, field

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import VipLlavaForConditionalGeneration
from transformers import AutoConfig 
from transformers import BitsAndBytesConfig

import torch

import traceback
import types

from typing import Optional, Callable

@dataclass
class ImageLMActionConfig:
    """
    Configuration for an ImageLMAction.

    The options in this object control what Hugging Face model is used,
    how the node interacts with the blackboard, and what decoding strategy
    is used. If you want a refresher on decoding strategies, check out 
    this blog post: https://huggingface.co/blog/how-to-generate.

    Args:
        model_name (`str`):
            The name of the model to use. This should be a valid name
            corresponding to a Hugging Face model name (including the user
            name).
        auto_load (`Optional[bool]`):
            An optional boolean indicating whether or not to automatically 
            load model either from disk or the Hugging Face hub. If `False`,
            the user is responsible for ensuring that a model is loaded
            before the first `tick()` is triggered. Defaults to `True`.
        text_input_key (`Optional[str]`):
            The blackboard key to use for writing and reading the text prompt 
            that this node will consume. Defaults to "text_in".
        image_input_key (`Optional[str]`):
            The blackboard key to use for writing and reading the image prompt
            that this node will consume. Defaults to "image_in".
        output_key (`Optional[str]`):
            The blackboard key to use for writing and reading the text generated
            by this node. Defaults to "out".
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
        max_new_tokens (`Optional[int]`):
            A limit on the number of new tokens to generate. You will usually want
            to set this yourself based on your application. Defaults to 16.
        do_sample (`Optional[bool]`):
            Optional boolean to control decoding strategy. If set to true, allows use
            of non-default generation strategy. Defaults to `False`.
        top_p (`Optional[float]`):
            Optional float to control use of nucleus sampling. If the value is strictly
            between 0 and 1, nucleus sampling is activated.
        torch_dtype (`torch.dtype`):
            The dtype to use for torch tensors. Defaults to `torch.float16`. You may
            need to change this depending on your quantization choices.
        use_flash_attn_2 (`Optional[bool]`):
            Optional bool controlling whether or not to use Flash Attention 2. Defaults
            to `False` in case you haven't installed flash attention. Substantially
            speeds up inference. 
    """
    model_name : str
    auto_load : Optional[bool] = field(
        default = True
    )
    text_input_key : Optional[str] = field(
        default = "text_in"
    )
    image_input_key : Optional[str] = field(
        default = "image_in"
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
    torch_dtype : Optional[torch.dtype] = field(
        default = torch.float16
    )
    use_flash_attn_2 : Optional[bool] = field(
        default = False
    )

class ImageLMAction(ActionNode):
    """
    An action node that uses a vision-language model to generate some
    text based on an image prompt and a text prompt contained in the 
    model's blackboard.

    This node is based on the Hugging Face transformers library, and will
    download the model that you specify by name. This can take a long 
    time and/or use a lot of storage, depending on the model you name.

    There are enough configuration options for this type of node that
    the options have all been placed in a dataclass config object. See 
    the documentation for that object to learn about the many options
    available to you.

    Args:
        cfg (`ImageLMActionConfig`):
            The configuration object for this model.
    """
    def __init__(self, name : str, cfg : ImageLMActionConfig) -> None:
        super().__init__(name)

        self.supported_model_map = {
            "llava" : LlavaForConditionalGeneration,
            "vipllava" : VipLlavaForConditionalGeneration
        }

        self.text_input_key = cfg.text_input_key
        self.image_input_key = cfg.image_input_key
        self.output_key = cfg.output_key
        self.device = cfg.device

        self.max_new_tokens = cfg.max_new_tokens
        self.do_sample = cfg.do_sample
        self.top_p = cfg.top_p

        self.torch_dtype = cfg.torch_dtype

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
            hf_config = AutoConfig.from_pretrained(cfg.model_name)
            model_type = hf_config.model_type

            try:
                ClassRef = self.supported_model_map[model_type]
                self.model = ClassRef.from_pretrained(
                    cfg.model_name,
                    low_cpu_mem_usage=True,
                    attn_implementation=self.attn_implementation,
                    quantization_config=self.bnb_cfg
                )
                self.model.eval()
            except Exception as e:
                print(f"Failed to create model of type {model_type}")
                self.model = None

            self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        else:
            self.model = None
            self.processor = None
        
        self.input_processor = None
        self.output_processor = None

    def set_model(self, new_model) -> None:
        """
        Set a new model to use for generating text.
        """
        self.model = new_model
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(new_model.name_or_path)

    def set_input_processor(self, f : Callable) -> None:
        """
        Set the input processor to use during `tick()`s. 

        An input processor is applied to the prompt image and the prompt text 
        stored in the blackboard, and can be used to preprocess the prompt. 
        The processor function should be a map from `str` to `str`. During 
        a `tick()`, the output of this function will be what is tokenized 
        and sent to the model for generation.

        Args:
            f (`Callable`):
                The input processor function to use. Should be a callable
                object that maps (image,string) pairs to (image,string) 
                pairs.
        """        
        self.input_processor = types.MethodType(f, self)

    def set_output_processor(self, f : Callable) -> None:
        """
        Set the output processor to use during `tick()`s.

        An output processor is applied to the text generated by the model,
        before that text is written to the output slot of the blackboard.
        The function should be a map from `str` to `str`.

        A typical example of an output processor would be a function that
        removes the prompt from the text returned by a model, so that only
        the newly generated text is written to the blackboard.

        Args:
            f (`Callable`):
                The output processor function. Should be a callable object
                that maps strings to strings.
        """
        self.output_processor = types.MethodType(f, self)

    def tick(self) -> NodeStatus:
        """
        Execute a tick, consisting of the following steps:

        - Retrieve the text prompt and image prompt for the node's blackboard.
        - Apply the input processor, if one exists,
        - Process the input text and image into ids for the model.
        - Generate new tokens based on the processed prompt.
        - Decode the model output into a text string.
        - Apply the output processor, if one exists.
        - Write the output text to the blackboard.

        If any of the above fail, the exception text is printed and the node
        returns a status of `FAILURE`. Otherwise the node returns `SUCCESS`.
        """
        try:
            input_text = self.blackboard[self.text_input_key]
            input_image = self.blackboard[self.image_input_key]

            if self.input_processor:
                input_text, input_image = self.input_processor(input_text, input_image)

            input_ids = self.processor(text=input_text, images=input_image, return_tensors="pt").to(self.model.device, self.torch_dtype)
            generated_ids = self.model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, top_p=self.top_p)
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if self.output_processor:
                output_text = self.output_processor(output_text)

            self.blackboard[self.output_key] = output_text

            return NodeStatus.SUCCESS
        except Exception as ex:
            print(f"Exception in node {self.name}:")
            print(traceback.format_exc())
            
            return NodeStatus.FAILURE