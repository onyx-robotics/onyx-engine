from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
import torch
# import jax
# import jax.numpy as jnp

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
        
        # Initialize scaling tensors
        # Inputs
        self.in_scale = torch.ones(n_inputs, dtype=dtype, device=device)
        self.in_bias = torch.zeros(n_inputs, dtype=dtype, device=device)
        self.in_unscale = torch.ones(n_inputs, dtype=dtype, device=device)
        self.in_unbias = torch.zeros(n_inputs, dtype=dtype, device=device)
        # Outputs
        self.out_scale = torch.ones(n_outputs, dtype=dtype, device=device)
        self.out_bias = torch.zeros(n_outputs, dtype=dtype, device=device)
        self.out_unscale = torch.ones(n_outputs, dtype=dtype, device=device)
        self.out_unbias = torch.zeros(n_outputs, dtype=dtype, device=device)

        # Helper function to compute scaling factors
        def compute_scale_factors(feature: Union[Input, Output, State], scale_tensor, bias_tensor, unscale_tensor, unbias_tensor, idx):
            if feature.scale is None:
                return scale_tensor, bias_tensor, unscale_tensor, unbias_tensor
            
            if feature.scale == 'mean':
                mean = feature.train_mean or 0.0
                std = feature.train_std or 1.0
                # Scaling: x_norm = (x - mean) / std
                scale_tensor[idx] = 1.0 / std
                bias_tensor[idx] = -mean / std
                # Descaling: x = x_norm * std + mean
                unscale_tensor[idx] = std
                unbias_tensor[idx] = mean
            else:
                min_val = feature.scale[0] or 0.0
                max_val = feature.scale[1] or 1.0
                scale_range = max_val - min_val
                # Scaling: x_norm = 2 * (x - min) / (max - min) - 1
                scale_tensor[idx] = 2.0 / scale_range
                bias_tensor[idx] = -1.0 - (2.0 * min_val / scale_range)
                # Descaling: x = 0.5 * (x_norm + 1) * (max - min) + min
                unscale_tensor[idx] = 0.5 * scale_range
                unbias_tensor[idx] = (0.5 * scale_range) + min_val

        # Compute scaling factors for inputs and outputs
        for i, feature in enumerate(inputs):
            compute_scale_factors(feature, self.in_scale, self.in_bias, 
                                self.in_unscale, self.in_unbias, i)
            
        for i, feature in enumerate(outputs):
            compute_scale_factors(feature, self.out_scale, self.out_bias,
                                self.out_unscale, self.out_unbias, i)

        # Reshape tensors for efficient broadcasting
        # Input tensors: (1, 1, n_features) for batch, sequence, feature dimensions
        self.in_scale = self.in_scale.view(1, 1, -1)
        self.in_bias = self.in_bias.view(1, 1, -1)
        self.in_unscale = self.in_unscale.view(1, 1, -1)
        self.in_unbias = self.in_unbias.view(1, 1, -1)
        
        # Output tensors: (1, n_features) for batch, feature dimensions
        self.out_scale = self.out_scale.view(1, -1)
        self.out_bias = self.out_bias.view(1, -1)
        self.out_unscale = self.out_unscale.view(1, -1)
        self.out_unbias = self.out_unbias.view(1, -1)

    def set_device(self, device: torch.device):
        self.device = device
        self.in_scale = self.in_scale.to(device)
        self.in_bias = self.in_bias.to(device)
        self.in_unscale = self.in_unscale.to(device)
        self.in_unbias = self.in_unbias.to(device)
        self.out_scale = self.out_scale.to(device)
        self.out_bias = self.out_bias.to(device)
        self.out_unscale = self.out_unscale.to(device)
        self.out_unbias = self.out_unbias.to(device)

    def scale_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input features to normalized range. (most common)"""
        if self.device != x.device:
            self.set_device(x.device)
        return x * self.in_scale + self.in_bias

    def unscale_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Unscale input features back to original scale."""
        if self.device != x.device:
            self.set_device(x.device)
        return x * self.in_unscale + self.in_unbias

    def scale_outputs(self, y: torch.Tensor) -> torch.Tensor:
        """Scale output features to normalized range."""
        if self.device != y.device:
            self.set_device(y.device)
        return y * self.out_scale + self.out_bias

    def unscale_outputs(self, y: torch.Tensor) -> torch.Tensor:
        """Unscale output features back to original scale. (most common)"""
        if self.device != y.device:
            self.set_device(y.device)
        return y * self.out_unscale + self.out_unbias
    
# class FeatureScalerJax:
#     def __init__(self, outputs: List[Output], inputs: List[Union[Input, State]], device: jax.Device = None, dtype=jax.numpy.float32):
#         self.device = device
#         self.dtype = dtype
#         n_inputs = len(inputs)
#         n_outputs = len(outputs)
        
#         # Initialize scaling arrays
#         # Inputs
#         self.in_scale = jnp.ones(n_inputs, dtype=dtype)
#         self.in_bias = jnp.zeros(n_inputs, dtype=dtype)
#         self.in_unscale = jnp.ones(n_inputs, dtype=dtype)
#         self.in_unbias = jnp.zeros(n_inputs, dtype=dtype)
#         # Outputs
#         self.out_scale = jnp.ones(n_outputs, dtype=dtype)
#         self.out_bias = jnp.zeros(n_outputs, dtype=dtype)
#         self.out_unscale = jnp.ones(n_outputs, dtype=dtype)
#         self.out_unbias = jnp.zeros(n_outputs, dtype=dtype)

#         # Helper function to compute scaling factors
#         def compute_scale_factors(feature: Union[Input, Output, State], scale_array, bias_array, unscale_array, unbias_array, idx):
#             if feature.scale is None:
#                 # Return arrays unchanged when no scaling is needed
#                 return scale_array, bias_array, unscale_array, unbias_array
            
#             if feature.scale == 'mean':
#                 mean = feature.train_mean or 0.0
#                 std = feature.train_std or 1.0
#                 # Scaling: x_norm = (x - mean) / std
#                 scale_array = scale_array.at[idx].set(1.0 / std)
#                 bias_array = bias_array.at[idx].set(-mean / std)
#                 # Descaling: x = x_norm * std + mean
#                 unscale_array = unscale_array.at[idx].set(std)
#                 unbias_array = unbias_array.at[idx].set(mean)
#             else:
#                 min_val = feature.scale[0] or 0.0
#                 max_val = feature.scale[1] or 1.0
#                 scale_range = max_val - min_val
#                 # Scaling: x_norm = 2 * (x - min) / (max - min) - 1
#                 scale_array = scale_array.at[idx].set(2.0 / scale_range)
#                 bias_array = bias_array.at[idx].set(-1.0 - (2.0 * min_val / scale_range))
#                 # Descaling: x = 0.5 * (x_norm + 1) * (max - min) + min
#                 unscale_array = unscale_array.at[idx].set(0.5 * scale_range)
#                 unbias_array = unbias_array.at[idx].set((0.5 * scale_range) + min_val)
            
#             return scale_array, bias_array, unscale_array, unbias_array

#         # Compute scaling factors for inputs and outputs
#         for i, feature in enumerate(inputs):
#             self.in_scale, self.in_bias, self.in_unscale, self.in_unbias = compute_scale_factors(
#                 feature, self.in_scale, self.in_bias, self.in_unscale, self.in_unbias, i)
            
#         for i, feature in enumerate(outputs):
#             self.out_scale, self.out_bias, self.out_unscale, self.out_unbias = compute_scale_factors(
#                 feature, self.out_scale, self.out_bias, self.out_unscale, self.out_unbias, i)

#         # Reshape arrays for efficient broadcasting
#         # Input arrays: (1, 1, n_features) for batch, sequence, feature dimensions
#         self.in_scale = self.in_scale.reshape(1, 1, -1)
#         self.in_bias = self.in_bias.reshape(1, 1, -1)
#         self.in_unscale = self.in_unscale.reshape(1, 1, -1)
#         self.in_unbias = self.in_unbias.reshape(1, 1, -1)
        
#         # Output arrays: (1, n_features) for batch, feature dimensions
#         self.out_scale = self.out_scale.reshape(1, -1)
#         self.out_bias = self.out_bias.reshape(1, -1)
#         self.out_unscale = self.out_unscale.reshape(1, -1)
#         self.out_unbias = self.out_unbias.reshape(1, -1)

#     def set_device(self, device: jax.Device):
#         """Set the device for all scaling arrays.
        
#         Note: In JAX, arrays are typically placed on devices automatically based on context.
#         This method is provided for explicit device control when needed.
#         """
#         self.device = device
#         # Use jax.device_put to move arrays to the specified device
#         self.in_scale = jax.device_put(self.in_scale, device)
#         self.in_bias = jax.device_put(self.in_bias, device)
#         self.in_unscale = jax.device_put(self.in_unscale, device)
#         self.in_unbias = jax.device_put(self.in_unbias, device)
#         self.out_scale = jax.device_put(self.out_scale, device)
#         self.out_bias = jax.device_put(self.out_bias, device)
#         self.out_unscale = jax.device_put(self.out_unscale, device)
#         self.out_unbias = jax.device_put(self.out_unbias, device)

#     def scale_inputs(self, x: jax.Array) -> jax.Array:
#         """Scale input features to normalized range. (most common)"""
#         return x * self.in_scale + self.in_bias

#     def unscale_inputs(self, x: jax.Array) -> jax.Array:
#         """Unscale input features back to original scale."""
#         return x * self.in_unscale + self.in_unbias

#     def scale_outputs(self, y: jax.Array) -> jax.Array:
#         """Scale output features to normalized range."""
#         return y * self.out_scale + self.out_bias

#     def unscale_outputs(self, y: jax.Array) -> jax.Array:
#         """Unscale output features back to original scale. (most common)"""
#         return y * self.out_unscale + self.out_unbias
