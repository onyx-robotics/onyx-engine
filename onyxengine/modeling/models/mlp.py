import torch.nn as nn
from pydantic import BaseModel, Field
from typing import Literal
from onyxengine.modeling import ModelSimulatorConfig, ModelSimulator

class MLPConfig(BaseModel):
    onyx_model_type: str = Field(default='mlp', frozen=True, init=False)
    sim_config: ModelSimulatorConfig = ModelSimulatorConfig()
    num_inputs: int = 1
    num_outputs: int = 1
    sequence_length: int = 1
    hidden_layers: int = 2
    hidden_size: int = 32
    activation: Literal['relu', 'tanh', 'sigmoid'] = 'relu'
    dropout: float = 0.0
    bias: bool = True

class MLP(nn.Module, ModelSimulator):
    def __init__(self, config: MLPConfig):
        nn.Module.__init__(self)
        ModelSimulator.__init__(self, config.sim_config)
        self.config = config
        num_inputs = config.num_inputs * config.sequence_length
        num_outputs = config.num_outputs
        hidden_layers = config.hidden_layers
        hidden_size = config.hidden_size
        activation = None
        if config.activation == 'relu':
            activation = nn.ReLU()
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
        layers.append(activation)
        layers.append(nn.Dropout(dropout))
        
        # Add remaining hidden layers
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
        
        # Add output layer
        layers.append(nn.Linear(hidden_size, num_outputs, bias=bias))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights close to zero
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.01, b=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Sequence input shape (batch_size, sequence_length, num_inputs)
        return self.model(x.view(x.size(0), -1))