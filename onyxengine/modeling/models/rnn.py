import torch
import torch.nn as nn
from pydantic import Field, model_validator
from typing_extensions import Self
from typing import Literal, List, Union, Dict
from onyxengine.modeling import (
    OnyxModelBaseConfig,
    OnyxModelOptBaseConfig,
    validate_param,
    validate_opt_param,
    ModelSimulator,
    FeatureScaler,
)    

class RNNConfig(OnyxModelBaseConfig):
    """
    Configuration class for the RNN model.
    
    Args:
        type (str): Model type = 'rnn', immutable.
        outputs (List[Output]): List of output variables.
        inputs (List[Input | State]): List of input variables.
        dt (float): Time step for the model.
        sequence_length (int): Length of input sequences (default is 1).
        rnn_type (Literal['RNN', 'LSTM', 'GRU']): Type of RNN to use (default is 'LSTM').
        hidden_layers (int): Number of hidden layers in the RNN (default is 2).
        hidden_size (int): Number of hidden units in each layer (default is 32).
        dropout (float): Dropout rate (default is 0.0).
        bias (bool): Whether or not to include bias in the RNN (default is True).
    """
    type: Literal['rnn'] = Field(default='rnn', frozen=True, init=False)
    rnn_type: Literal['RNN', 'LSTM', 'GRU'] = 'LSTM'
    hidden_layers: int = 2
    hidden_size: int = 32
    dropout: float = 0.0
    bias: bool = True
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_param(self.hidden_layers, 'hidden_layers', min_val=1, max_val=10)
        validate_param(self.hidden_size, 'hidden_size', min_val=1, max_val=1024)
        validate_param(self.dropout, 'dropout', min_val=0.0, max_val=1.0)
        return self
    
class RNNOptConfig(OnyxModelOptBaseConfig):
    """
    Optimization config for the RNN model.
    
    Args:
        type (str): Model type = 'rnn_opt', immutable.
        outputs (List[Output]): List of output variables.
        inputs (List[Input | State]): List of input variables.
        dt (float): Time step for the model.
        rnn_type (Union[Literal['RNN', 'LSTM', 'GRU'], Dict[str, List[str]]): Type of RNN to use (default is {"select": ['RNN', 'LSTM', 'GRU']}).
        sequence_length (Union[int, Dict[str, List[int]]): Length of input sequences (default is {"select": [1, 2, 4, 5, 6, 8, 10, 12, 14, 15]}).
        hidden_layers (Union[int, Dict[str, List[int]]): Number of hidden layers in the RNN (default is {"range": [2, 5, 1]}).
        hidden_size (Union[int, Dict[str, List[int]]): Number of hidden units in each layer (default is {"select": [12, 24, 32, 64, 128]}).
        dropout (Union[float, Dict[str, List[float]]): Dropout rate (default is {"range": [0.0, 0.4, 0.1]}).
        bias (Union[bool, Dict[str, List[bool]]): Whether or not to include bias in the RNN (default is True).
    """
    type: Literal['rnn_opt'] = Field(default='rnn_opt', frozen=True, init=False)
    rnn_type: Union[Literal['RNN', 'LSTM', 'GRU'], Dict[str, List[str]]] = {"select": ['RNN', 'LSTM', 'GRU']}
    hidden_layers: Union[int, Dict[str, List[int]]] = {"range": [2, 5, 1]}
    hidden_size: Union[int, Dict[str, List[int]]] = {"select": [12, 24, 32, 64, 128]}
    dropout: Union[float, Dict[str, List[float]]] = {"range": [0.0, 0.4, 0.1]}
    bias: Union[bool, Dict[str, List[bool]]] = True
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> Self:
        validate_opt_param(self.rnn_type, 'rnn_type', options=['select'], select_from=['RNN', 'LSTM', 'GRU'])
        validate_opt_param(self.hidden_layers, 'hidden_layers', options=['select', 'range'], min_val=1, max_val=10)
        validate_opt_param(self.hidden_size, 'hidden_size', options=['select', 'range'], min_val=1, max_val=1024)
        validate_opt_param(self.dropout, 'dropout', options=['select', 'range'], min_val=0.0, max_val=1.0)
        validate_opt_param(self.bias, 'bias', options=['select'], select_from=[True, False])
        return self

class RNN(nn.Module, ModelSimulator):
    def __init__(self, config: RNNConfig):
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
        self.rnn_type = config.rnn_type
        num_inputs = len(config.inputs)
        num_outputs = len(config.outputs)
        self.sequence_length = config.sequence_length
        self.hidden_layers = config.hidden_layers
        self.hidden_size = config.hidden_size
        
        # Build the RNN
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(num_inputs, self.hidden_size, self.hidden_layers, dropout=config.dropout, bias=config.bias, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(num_inputs, self.hidden_size, self.hidden_layers, dropout=config.dropout, bias=config.bias, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(num_inputs, self.hidden_size, self.hidden_layers, dropout=config.dropout, bias=config.bias, batch_first=True)
        else:
            raise ValueError("Invalid RNN type. Choose from 'RNN', 'LSTM', or 'GRU'.")
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, num_outputs)
        
    def forward(self, x):
        # Init the hidden state
        batch_size = x.size(0)
        if self.rnn_type == 'LSTM':
            hidden_state = (torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=x.device),
                        torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=x.device))
        else:
            hidden_state = torch.zeros(self.hidden_layers, batch_size, self.hidden_size, device=x.device)
                
        x = self.feature_scaler.scale_inputs(x)
        rnn_output, _ = self.rnn(x, hidden_state)
        normalized_output = self.layer_norm(rnn_output[:, -1, :])
        network_output = self.output_layer(normalized_output)
        return self.feature_scaler.unscale_outputs(network_output)