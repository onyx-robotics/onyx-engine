import torch
import torch.nn as nn
from pydantic import Field, model_validator
from typing_extensions import Self
from typing import Literal, List, Union, Dict
# from flax import nnx
# import jax.numpy as jnp
from onyxengine.modeling import (
    OnyxModelBaseConfig,
    OnyxModelOptBaseConfig,
    validate_param,
    validate_opt_param,
    ModelSimulator,
    FeatureScaler,
    # FeatureScalerJax
)

class MLPConfig(OnyxModelBaseConfig):
    """
    Configuration class for the MLP model.

    Args:
        type (str): Model type = 'mlp', immutable.
        outputs (List[Output]): List of output variables.
        inputs (List[Input | State]): List of input variables.
        dt (float): Time step for the model.
        sequence_length (int): Length of the input sequence (default is 1).
        hidden_layers (int): Number of hidden layers (default is 2).
        hidden_size (int): Size of each hidden layer (default is 32).
        activation (Literal['relu', 'gelu', 'tanh', 'sigmoid']): Activation function (default is 'relu').
        dropout (float): Dropout rate for layers (default is 0.0).
        bias (bool): Whether to use bias in layers (default is True).
    """
    type: Literal['mlp'] = Field(default='mlp', frozen=True, init=False)
    hidden_layers: int = 2
    hidden_size: int = 32
    activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = 'relu'
    dropout: float = 0.0
    bias: bool = True
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.hidden_layers, 'hidden_layers', min_val=1, max_val=10)
        validate_param(self.hidden_size, 'hidden_size', min_val=1, max_val=1024)
        validate_param(self.dropout, 'dropout', min_val=0.0, max_val=1.0)
        return self

class MLPOptConfig(OnyxModelOptBaseConfig):
    """
    Optimization config class for the MLP model.
    
    Args:
        type (str): Model type = 'mlp_opt', immutable.
        outputs (List[Output]): List of output variables.
        inputs (List[Input | State]): List of input variables.
        dt (float): Time step for the model.
        sequence_length (Union[int, Dict[str, List[int]]): Length of the input sequence (default is {"select": [1, 2, 4, 5, 6, 8, 10]}).
        hidden_layers (Union[int, Dict[str, List[int]]): Number of hidden layers (default is {"range": [2, 5, 1]}).
        hidden_size (Union[int, Dict[str, List[int]]): Size of each hidden layer (default is {"select": [12, 24, 32, 64, 128]}).
        activation (Union[Literal['relu', 'gelu', 'tanh', 'sigmoid'], Dict[str, List[str]]): Activation function (default is {"select": ['relu', 'gelu', 'tanh']}).
        dropout (Union[float, Dict[str, List[float]]): Dropout rate for layers (default is {"range": [0.0, 0.4, 0.1]}).
        bias (Union[bool, Dict[str, List[bool]]): Whether to use bias in layers (default is True).
    """
    type: Literal['mlp_opt'] = Field(default='mlp_opt', frozen=True, init=False)
    hidden_layers: Union[int, Dict[str, List[int]]] = {"range": [2, 5, 1]}
    hidden_size: Union[int, Dict[str, List[int]]] = {"select": [12, 24, 32, 64, 128]}
    activation: Union[Literal['relu', 'gelu', 'tanh', 'sigmoid'], Dict[str, List[str]]] = {"select": ['relu', 'gelu', 'tanh']}
    dropout: Union[float, Dict[str, List[float]]] = {"range": [0.0, 0.4, 0.1]}
    bias: Union[bool, Dict[str, List[bool]]] = True
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_opt_param(self.hidden_layers, 'hidden_layers', options=['select', 'range'], min_val=1, max_val=10)
        validate_opt_param(self.hidden_size, 'hidden_size', options=['select', 'range'], min_val=1, max_val=1024)
        validate_opt_param(self.activation, 'activation', options=['select'], select_from=['relu', 'gelu', 'tanh', 'sigmoid'])
        validate_opt_param(self.dropout, 'dropout', options=['select', 'range'], min_val=0.0, max_val=1.0)
        validate_opt_param(self.bias, 'bias', options=['select'], select_from=[True, False])
        return self


class MLP(nn.Module, ModelSimulator):
    def __init__(self, config: MLPConfig):
        nn.Module.__init__(self)
        ModelSimulator.__init__(
            self,
            outputs=config.outputs,
            inputs=config.inputs,
            sequence_length=config.sequence_length,
            dt=config.dt,
        )
        self.feature_scaler = FeatureScaler(outputs=config.outputs, inputs=config.inputs)
        self.config = config
        num_inputs = len(config.inputs) * config.sequence_length
        num_outputs = len(config.outputs)
        hidden_layers = config.hidden_layers
        hidden_size = config.hidden_size
        activation = None
        if config.activation == 'relu':
            activation = nn.ReLU()
        elif config.activation == 'gelu':
            activation = nn.GELU()
        elif config.activation == 'tanh':
            activation = nn.Tanh()
        elif config.activation == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {config.activation} not supported")
        dropout = config.dropout
        bias = config.bias
        layers = []
        
        # Add first hidden layer
        layers.append(nn.Linear(num_inputs, hidden_size, bias=bias))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation)
        layers.append(nn.Dropout(dropout))
        
        # Add remaining hidden layers
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
        
        # Add output layer
        layers.append(nn.Linear(hidden_size, num_outputs, bias=bias))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # Sequence input shape (batch_size, sequence_length, num_inputs)
        # Flatten to (batch_size, sequence_length * num_inputs)
        x = self.feature_scaler.scale_inputs(x)
        x = x.view(x.size(0), -1)
        return self.feature_scaler.unscale_outputs(self.model(x))
    
# class MLPJax(nnx.Module):
#     def __init__(self, config: MLPConfig):
#         nnx.Module.__init__(self)
#         # ModelSimulator.__init__(
#         #     self,
#         #     outputs=config.outputs,
#         #     inputs=config.inputs,
#         #     sequence_length=config.sequence_length,
#         #     dt=config.dt,
#         # )
#         self.feature_scaler = FeatureScalerJax(outputs=config.outputs, inputs=config.inputs)
#         self.config = config
#         num_inputs = len(config.inputs) * config.sequence_length
#         num_outputs = len(config.outputs)
#         hidden_layers = config.hidden_layers
#         hidden_size = config.hidden_size
#         activation = None
#         if config.activation == 'relu':
#             activation = nnx.relu
#         elif config.activation == 'gelu':
#             activation = nnx.gelu
#         elif config.activation == 'tanh':
#             activation = nnx.tanh
#         elif config.activation == 'sigmoid':
#             activation = nnx.sigmoid
#         else:
#             raise ValueError(f"Activation function {config.activation} not supported")
#         dropout = config.dropout
#         bias = config.bias
#         rngs = nnx.Rngs(0)
#         self.layers = []
        
#         # Add first hidden layer
#         self.layers.append(nnx.Linear(num_inputs, hidden_size, use_bias=bias, rngs=rngs))
#         self.layers.append(nnx.LayerNorm(hidden_size, rngs=rngs))
#         self.layers.append(activation)
#         self.layers.append(nnx.Dropout(dropout, rngs=rngs))
        
#         # Add remaining hidden layers
#         for _ in range(hidden_layers - 1):
#             self.layers.append(nnx.Linear(hidden_size, hidden_size, use_bias=bias, rngs=rngs))
#             self.layers.append(nnx.LayerNorm(hidden_size, rngs=rngs))
#             self.layers.append(activation)
#             self.layers.append(nnx.Dropout(dropout, rngs=rngs))
        
#         # Add output layer
#         self.layers.append(nnx.Linear(hidden_size, num_outputs, use_bias=bias, rngs=rngs))
            
#     def __call__(self, x):
#         # Sequence input shape (batch_size, sequence_length, num_inputs)
#         # Flatten to (batch_size, sequence_length * num_inputs)
#         x = self.feature_scaler.scale_inputs(x)
#         x = x.reshape(x.shape[0], -1)
        
#         for layer in self.layers:
#             x = layer(x)
            
#         return self.feature_scaler.unscale_outputs(x)
        
#     def load_from_pt(self, pt_path: str):
#         """
#         Load weights from a PyTorch .pt file and convert them to Flax format.
        
#         Args:
#             pytorch_model_path (str): Path to the PyTorch model file (.pt)
#         """
#         # Load PyTorch model state dict
#         state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)

#         # Process each layer in the Sequential model
#         for i, layer in enumerate(self.layers):
#             if isinstance(layer, nnx.Linear):
#                 # This is a linear layer, check if we have weights for it
#                 weight_key = f'model.{i}.weight'
#                 bias_key = f'model.{i}.bias'
                
#                 if weight_key in state_dict:
#                     # Convert PyTorch weight to JAX format
#                     pytorch_weight = state_dict[weight_key]
#                     # PyTorch uses (out_features, in_features) while Flax uses (in_features, out_features)
#                     jax_weight = jnp.array(pytorch_weight.detach().numpy().T)
                    
#                     # Update the layer's kernel using the proper method
#                     layer.kernel.value = jax_weight
#                     # print(f"Updated layer {i} kernel")
                
#                 if bias_key in state_dict and self.config.bias:
#                     # Convert PyTorch bias to JAX format
#                     pytorch_bias = state_dict[bias_key]
#                     jax_bias = jnp.array(pytorch_bias.detach().numpy())
                    
#                     # Update the layer's bias using the proper method
#                     layer.bias.value = jax_bias
#                     # print(f"Updated layer {i} bias")
                
#             elif isinstance(layer, nnx.LayerNorm):
#                 # LayerNorm has scale and bias parameters
#                 scale_key = f'model.{i}.weight'  # PyTorch LayerNorm weight
#                 bias_key = f'model.{i}.bias'     # PyTorch LayerNorm bias
                
#                 if scale_key in state_dict:
#                     pytorch_scale = state_dict[scale_key]
#                     jax_scale = jnp.array(pytorch_scale.detach().numpy())
#                     layer.scale.value = jax_scale
#                     # print(f"Updated layer {i} scale")
                
#                 if bias_key in state_dict:
#                     pytorch_bias = state_dict[bias_key]
#                     jax_bias = jnp.array(pytorch_bias.detach().numpy())
#                     layer.bias.value = jax_bias
#                     # print(f"Updated layer {i} bias")
                
#             # Skip activation functions and dropout layers as they don't have parameters
#             elif callable(layer) or isinstance(layer, nnx.Dropout):
#                 pass