from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
import torch

class BaseFeature(BaseModel):
    type: Literal['base_feature'] = Field(default='base_feature', frozen=True, init=False)
    name: str
    scale: Union[None, Literal['mean'], List[float]] = 'mean'
    train_mean: Optional[float] = Field(default=None, init=False)
    train_std: Optional[float] = Field(default=None, init=False)
    train_min: Optional[float] = Field(default=None, init=False)
    train_max: Optional[float] = Field(default=None, init=False)
    
    @model_validator(mode='after')
    def validate_scale(self) -> Self:
        if isinstance(self.scale, list):
            if len(self.scale) != 2:
                raise ValueError("Scale list must have 2 values representing the range of real-world values for this feature as: [min, max]")
            if self.scale[0] >= self.scale[1]:
                raise ValueError("Scale must be in the form [min, max] where min < max")
            
        return self

class Output(BaseFeature):
    """
    A standard output feature to be used by the model.
    
    Args:
        name (str): Name of the output feature.
        scale (Union[None, Literal['mean'], List[float]]): Scale for the output feature:
            
            - None: Feature is not scaled.
            - 'mean': Feature is scaled to have a mean of 0 and std of 1. (Default).
            - List[float]: Feature is scaled from its real-world [min, max] to a range of [-1, 1].
    """
    
    type: Literal['output'] = Field(default='output', frozen=True, init=False)
    
class Input(BaseFeature):
    """
    A standard input feature to be used by the model.
    
    Args:
        name (str): Name of the input feature.
        scale (Union[None, Literal['mean'], List[float]]): Scale for the output feature:
            
            - None: Feature is not scaled.
            - 'mean': Feature is scaled to have a mean of 0 and std of 1. (Default).
            - List[float]: Feature is scaled from its real-world [min, max] to a range of [-1, 1].
    """
    
    type: Literal['input'] = Field(default='input', frozen=True, init=False)
    
class State(BaseFeature):
    """
    A state feature that can be derived from a parent feature through different relationships (output, delta, or derivative).

    Args:
        name (str): Name of the state feature.
        relation (Literal['output', 'delta', 'derivative']): Method to solve for the feature:
        
            - 'output': Feature is the direct output of the model
            - 'delta': Feature is the change/delta of the parent value
            - 'derivative': Feature is the derivative of the parent value
        parent (str): Name of the parent feature from which this state is derived
        scale (Union[None, Literal['mean'], List[float]]): Scale for the output feature:
            
            - None: Feature is not scaled.
            - 'mean': Feature is scaled to have a mean of 0 and std of 1. (Default).
            - List[float]: Feature is scaled from its real-world [min, max] to a range of [-1, 1].
    """
    type: Literal['state'] = Field(default='state', frozen=True, init=False)
    relation: Literal['output', 'delta', 'derivative'] # Method to solve for the feature: the output of the model, parent is the delta of the value, or derivative of parent value
    parent: str # Parent feature to derive from
    
class Feature(BaseModel):
    config: Union[Input, Output, State] = Field(..., discriminator='type')
    
class FeatureScaler:
    def __init__(self, outputs: List[Output], inputs: List[Union[Input, State]], device: torch.device = torch.device('cpu'), dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        n_inputs = len(inputs)
        n_outputs = len(outputs)
        
        # Pre-compute the scaling factors for the inputs
        self.input_scale = torch.ones(n_inputs, dtype=dtype, device=device)
        self.input_bias = torch.zeros(n_inputs, dtype=dtype, device=device)
        for i, feature in enumerate(inputs):
            if feature.scale is None:
                continue  # Keep scale=1 and bias=0 for no scaling
            elif feature.scale == 'mean':
                # Formula: x_norm = (x - mean) / std
                mean = feature.train_mean or 0.0
                std = feature.train_std or 1.0
                self.input_scale[i] = 1.0 / std
                self.input_bias[i] = -mean / std
            else:
                # Formula: x_norm = 2 * (x - min) / (max - min) - 1
                min_val = feature.scale[0] or 0.0
                max_val = feature.scale[1] or 1.0
                scale_range = max_val - min_val
                self.input_scale[i] = 2.0 / scale_range
                self.input_bias[i] = -1.0 - (2.0 * min_val / scale_range)
                
        # Pre-compute the scaling factors for the outputs
        self.output_scale = torch.ones(n_outputs, dtype=dtype, device=device)
        self.output_bias = torch.zeros(n_outputs, dtype=dtype, device=device)
        for i, feature in enumerate(outputs):
            if feature.scale is None:
                continue  # Keep scale=1 and bias=0 for no scaling
            elif feature.scale == 'mean':
                # Formula: y = y_norm * std + mean
                self.output_scale[i] = feature.train_std or 1.0
                self.output_bias[i] = feature.train_mean or 0.0
            else:
                # Formula: y = 0.5 * (y_norm + 1) * (max - min) + min
                min_val = feature.scale[0] or 0.0
                max_val = feature.scale[1] or 1.0
                scale_range = max_val - min_val
                self.output_scale[i] = 0.5 * scale_range
                self.output_bias[i] = (0.5 * scale_range) + min_val

    def set_device(self, device: torch.device):
        self.device = device
        self.input_scale = self.input_scale.to(device)
        self.input_bias = self.input_bias.to(device)
        self.output_scale = self.output_scale.to(device)
        self.output_bias = self.output_bias.to(device)

    def scale_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.device != x.device:
            self.set_device(x.device)
        scale = self.input_scale.view(1, 1, -1)
        bias = self.input_bias.view(1, 1, -1)
        return x * scale + bias

    def descale_outputs(self, y: torch.Tensor) -> torch.Tensor:
        if self.device != y.device:
            self.set_device(y.device)
        scale = self.output_scale.view(1, -1)
        bias = self.output_bias.view(1, -1)
        return y * scale + bias