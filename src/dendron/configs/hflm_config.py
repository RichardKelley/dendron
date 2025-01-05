import os
from dataclasses import dataclass
from typing import Union

import torch
from transformers import PreTrainedModel

@dataclass
class HFLMConfig:
    _model: Union[str, PreTrainedModel]
    _device: str
    _parallelize: bool
    _dtype: Union[str, torch.dtype]
    _load_in_4bit: bool
    _load_in_8bit: bool
    _add_bos_token: bool
    _offload_folder: Union[str, os.PathLike]

    def __init__(self, 
                 model: Union[str, PreTrainedModel], 
                 device: str = "cuda",
                 parallelize: bool = False, 
                 dtype: Union[str, torch.dtype] = torch.float16, 
                 load_in_4bit: bool = False, 
                 load_in_8bit: bool = False, 
                 add_bos_token: bool = False, 
                 offload_folder: Union[str, os.PathLike] = None):
        self._model = model
        self._device = device
        self._parallelize = parallelize
        self._dtype = dtype
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._add_bos_token = add_bos_token
        self._offload_folder = offload_folder

    @property
    def model_name(self) -> str:
        if isinstance(self._model, str):
            return self._model
        elif isinstance(self._model, PreTrainedModel):
            return self._model.name_or_path
    
    @property
    def model(self) -> Union[str, PreTrainedModel]:
        return self._model
        
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def parallelize(self) -> bool:
        return self._parallelize
    
    @property
    def dtype(self) -> Union[str, torch.dtype]:
        return self._dtype

    @property
    def load_in_4bit(self) -> bool:
        return self._load_in_4bit

    @property
    def load_in_8bit(self) -> bool:
        return self._load_in_8bit

    @property
    def add_bos_token(self) -> bool:
        return self._add_bos_token

    @property
    def offload_folder(self) -> Union[str, os.PathLike]:
        return self._offload_folder

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parallelize": self.parallelize,
            "dtype": self.dtype,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "add_bos_token": self.add_bos_token,
            "offload_folder": self.offload_folder
        }   